"""v15: 3-CDR finetune with PERIODIC RE-ANCHORING of the epitope.

Differences from v9 / v14:
  - v9:  fixed epitope = 21-residue list auto-detected from input PDB
         (single anchor held for the whole run)
  - v14: dynamic topk = at every step, for each CDR residue, pick top-K
         closest target residues (self-referential, loss can be 0 with
         no exploration)
  - v15 (this file): EM-style alternating optimization.  Every
         ``--chunk-size`` design steps, an E-step re-extracts the
         predicted binder-antigen interface from the distogram of the
         current step and sets that as the *fixed* epitope for the next
         chunk.  M-step = same v9-style fixed-epitope loss, just with a
         different fixed list each chunk.

Why this should work better than v14:
  - Within a chunk, the epitope loss has a stable target, so the
    gradient is informative (the standard v9 force pulling CDRs toward
    a known set of residues).
  - Across chunks, the anchor is updated to track the optimizer — if
    the binder has drifted to a new contact patch, the next chunk's
    anchor follows.
  - This breaks the v14 self-reference: the E-step is a no_grad op
    on a *current* prediction, not a loss side-effect.

Pinned: framework 95 positions, mutable: 3 CDRs (32 positions), starting
from v2 step050 (same as v9), lr=0.05, steps=60, chunk_size=4.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS, MUTABLE_TOKEN,
    compute_structure_losses, get_mid_points, build_pdb_prior,
    compute_epitope_loss,
)
from test_b5_pdb import setup_design  # noqa: E402

from design_b5_mps_v2 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    load_model, MODEL_PATH, ESMC_PATH, DEVICE, NUM_RES_TYPES, TOKENS,
    AA_TO_TOKEN, AA_DIMS, CYS_TOK, AA_FREQ, TEMP_MIN,
    SAMPLE_STEPS_FWD, N_LOOPS_FWD, W_AA_FREQ,
    pin_fixed_positions,
)

# v2 step050 starting point (the classic best)
V2_STEP050_FULL = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEA"
                   "VAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCA"
                   "ARVVTDSYQPIYKAPIRWGQGTLVTVS")


def extract_predicted_epitope(
    disto: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    threshold: float = 10.0,
    min_size: int = 4,
    max_size: int = 40,
) -> tuple[list[int], dict]:
    """E-step: pick target residues that are predicted to be in contact
    with the binder in the current distogram.

    For each target residue j, compute min over binder residues i of
    E[d(i, j)] from the distogram.  Target residues whose closest
    binder residue is within ``threshold`` Å become the new epitope.

    Returns:
      epi_list: sorted list of target token indices
      diag: diagnostics (raw count, distances, etc.)

    If the raw count is below ``min_size``, fill up to ``min_size`` with
    the closest residues that didn't quite make the cutoff (graceful
    degradation when the binder has drifted far).  Cap at ``max_size``
    by taking the closest ``max_size`` residues.
    """
    cross = disto[:, -binder_length:, :-binder_length, :]   # [B, L_b, T, n_bins]
    probs = torch.softmax(cross, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                  # [B, L_b, T]
    # min over binder residues → for each target residue, distance to
    # the closest binder residue
    min_per_target = e_dist.min(dim=1).values                # [B, T]
    min_per_target = min_per_target.mean(dim=0)              # [T]
    n_target = min_per_target.numel()
    within = (min_per_target < threshold).nonzero(as_tuple=True)[0]
    within_list = sorted(within.tolist())
    diag = {
        "n_within_threshold": len(within_list),
        "n_target": n_target,
        "min_dist_min": float(min_per_target.min().item()),
        "min_dist_median": float(min_per_target.median().item()),
        "n_added_below_min": 0,
        "n_capped_above_max": 0,
    }
    if len(within_list) < min_size:
        # Fill up to min_size with the next-closest residues
        sorted_idx = torch.argsort(min_per_target).tolist()
        extras = [j for j in sorted_idx if j not in set(within_list)]
        need = min_size - len(within_list)
        within_list = sorted(within_list + extras[:need])
        diag["n_added_below_min"] = min(need, len(extras))
    if len(within_list) > max_size:
        # Cap at max_size by taking the closest max_size
        idx_set = set(within_list)
        sorted_idx = torch.argsort(min_per_target).tolist()
        within_list = sorted([j for j in sorted_idx if j in idx_set][:max_size])
        diag["n_capped_above_max"] = max_size
    return within_list, diag


def run_design(steps: int = 60,
               lr: float = 0.05,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               chunk_size: int = 4,
               epi_threshold: float = 10.0,
               epi_min_size: int = 4,
               epi_max_size: int = 40,
               epitope_cutoff: float = 8.0,
               log_every: int = 4,
               snapshot_every: int = 4,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v15_reanchor_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V2_STEP050_FULL):
    print(f"=== v15: 3-CDR finetune with PERIODIC RE-ANCHORING "
          f"(chunk={chunk_size}, threshold={epi_threshold}Å, lr={lr}, "
          f"steps={steps}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=epitope_cutoff, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]

    target_len = len(target_seq)
    binder_len = len(binder_template)
    init_epi = list(setup["epitope_token_indices"])  # v9's 21-residue input-PDB epitope
    cdr = setup["cdr_indices"]
    if prior_bins is None:
        prior_bins = setup["prior_bins"]
    if prior_mask is None:
        prior_mask = setup["prior_mask"]

    epi = list(init_epi)  # current fixed epitope (mutable, replaced by E-step)

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, init epitope {len(epi)}", flush=True)
    print(f"  init epitope residues: {epi}", flush=True)
    print(f"  Init: H1={init_seq[25:35]}  H2={init_seq[54:60]}  "
          f"H3={init_seq[101:117]}", flush=True)

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}", flush=True)

    template_list = list(init_seq)
    for i in cdr:
        template_list[i] = MUTABLE_TOKEN
    v15_template = "".join(template_list)
    assert v15_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v15_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v15_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} "
          f"(H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)", flush=True)

    history = []
    snapshots = []
    reanchor_log = []  # each entry: {step, old_size, new_size, added, dropped, threshold, n_within}
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_inter = float("inf")
    best_seq_inter = ""
    best_step_inter = -1
    best_iptm = -1.0
    best_seq_iptm = ""
    best_step_iptm = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1

    init_cur = soft_to_hard_seq(soft_logits)
    print(f"\n  init CDR: {''.join(init_cur[i] for i in cdr)}", flush=True)
    assert init_cur == init_seq, f"init seq != provided init_seq: {init_cur} vs {init_seq}"

    print(f"\nDesigning {steps} steps with periodic re-anchor every "
          f"{chunk_size} steps ...", flush=True)
    header = (f"  {'step':>4}  {'total':>7}  {'epi':>6}  {'inter':>6}  "
              f"{'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  "
              f"{'|epi|':>5}  CDR_seq")
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)

        cur_seq = soft_to_hard_seq(soft_logits)
        cdr_set = set(cdr)
        n_cdr_diff = sum(1 for i in cdr_set if cur_seq[i] != init_seq[i])
        non_cdr = [i for i in range(len(cur_seq)) if i not in cdr_set]
        n_fw_diff = sum(1 for i in non_cdr if cur_seq[i] != init_seq[i])
        if n_fw_diff > 0 and step % log_every == 0:
            print(f"  WARNING: framework has {n_fw_diff} diffs at step {step}",
                  flush=True)

        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
                calculate_confidence=True,
            )
        disto_bf = out["distogram_logits"].float()
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)

        # === Epitope loss: v9-style fixed, but the epi list is updated
        # by the E-step (no_grad) every chunk_size steps.
        epi_loss = compute_epitope_loss(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            bin_distance=get_mid_points(64, 2.0, 22.0).to(disto.device),
            cutoff=epitope_cutoff,
        )

        all_losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            epitope_cutoff=epitope_cutoff,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        # Override epitope_loss with our current-ebi version (they
        # match by construction; this is just to keep the bookkeeping
        # honest if the chunk's epi happens to equal the call's epi).
        all_losses["epitope_loss"] = epi_loss
        B = disto.size(0)
        total = (LOSS_WEIGHTS["intra_contact"] * all_losses["intra_contact_loss"]
                 + LOSS_WEIGHTS["inter_contact"] * all_losses["inter_contact_loss"]
                 + LOSS_WEIGHTS["glob"] * all_losses["glob_loss"]
                 + w_epitope * epi_loss
                 + LOSS_WEIGHTS["structure_prior"] * all_losses["structure_prior_loss"])
        total = total + w_aa_freq * lm
        total = total.sum() / B

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step, "total": float(total.item()),
            "soft_epi": float(epi_loss.item()),
            "inter": float(all_losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "n_cdr_diff_from_init": n_cdr_diff,
            "epi_size": len(epi),
        }
        history.append(record)

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {len(epi):>5}  {cdr_seq}  "
                  f"[{elapsed:>5.0f}s]", flush=True)

        if total.item() < best_total:
            best_total = total.item()
            best_seq_total = cur_seq; best_step_total = step
        if all_losses["inter_contact_loss"].item() < best_inter:
            best_inter = all_losses["inter_contact_loss"].item()
            best_seq_inter = cur_seq; best_step_inter = step
        if diag["cdr_to_epitope_min"] < best_cdr_to_epi:
            best_cdr_to_epi = diag["cdr_to_epitope_min"]
            best_seq_epi = cur_seq; best_step_epi = step
        if iptm is not None and iptm > best_iptm:
            best_iptm = iptm; best_seq_iptm = cur_seq; best_step_iptm = step

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            if soft_logits.grad is None:
                print(f"  [WARN step {step}] grad is None, skipping", flush=True)
                continue
            with torch.no_grad():
                soft_logits.grad[fixed_mask] = 0.0
            g_norm = soft_logits.grad.norm().item()
            if step % log_every == 0:
                print(f"  [step {step}] grad_norm={g_norm:.4f}  "
                      f"max={soft_logits.grad.abs().max().item():.4f}  "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, v15_template)

        # === v15 key trick: re-anchor epitope every chunk_size steps ===
        if (step + 1) % chunk_size == 0 and step < steps:
            t0 = time.time()
            with torch.no_grad():
                new_epi, epi_diag = extract_predicted_epitope(
                    disto.detach(),
                    binder_length=binder_len,
                    bin_distance=get_mid_points(64, 2.0, 22.0).to(disto.device),
                    threshold=epi_threshold,
                    min_size=epi_min_size,
                    max_size=epi_max_size,
                )
            old_set = set(epi)
            new_set = set(new_epi)
            added = sorted(new_set - old_set)
            dropped = sorted(old_set - new_set)
            epi = new_epi
            reanchor_log.append({
                "step": step + 1,
                "old_size": len(old_set),
                "new_size": len(new_epi),
                "added": added,
                "dropped": dropped,
                "threshold": epi_threshold,
                "n_within": epi_diag["n_within_threshold"],
                "min_dist_min": epi_diag["min_dist_min"],
                "min_dist_median": epi_diag["min_dist_median"],
                "n_added_below_min": epi_diag["n_added_below_min"],
                "n_capped_above_max": epi_diag["n_capped_above_max"],
                "time_s": time.time() - t0,
            })
            if step % log_every == 0 or step < 8:
                print(f"  [reanchor @ step {step + 1}] "
                      f"|old|={len(old_set)} |new|={len(new_epi)} "
                      f"+{len(added)} -{len(dropped)} "
                      f"within<{epi_threshold}Å={epi_diag['n_within_threshold']} "
                      f"min_dist_min={epi_diag['min_dist_min']:.2f}Å "
                      f"median={epi_diag['min_dist_median']:.2f}Å "
                      f"[{time.time() - t0:.1f}s]", flush=True)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "h1": cdr_seq[:10],
                "h2": cdr_seq[10:16],
                "h3": cdr_seq[16:],
                "full_seq": cur_seq,
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "ptm": ptm, "iptm": iptm,
                "n_cdr_diff_from_init": n_cdr_diff,
                "epi_size": len(epi),
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_full_seq": init_seq,
                    "binder_len": binder_len,
                    "init_epitope": init_epi,
                    "config": {
                        "model": "ESMFold2-Fast (721M) for design",
                        "starting_from": "v2 step050",
                        "prior_source": "v9 prior (Full-predicted v2 step050 3D CA coords, "
                                        "averaged over 4 diffusion samples)",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "epitope_strategy": "periodic re-anchor (EM-style)",
                        "chunk_size": chunk_size,
                        "epi_threshold": epi_threshold,
                        "epi_min_size": epi_min_size,
                        "epi_max_size": epi_max_size,
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                    "reanchor_log": reanchor_log,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)",
          flush=True)
    print(f"  re-anchors:    {len(reanchor_log)}", flush=True)
    for r in reanchor_log:
        print(f"    step {r['step']:>3}  |epi| {r['old_size']:>2}→{r['new_size']:>2}  "
              f"+{len(r['added']):>2} -{len(r['dropped']):>2}  "
              f"within<{epi_threshold}Å={r['n_within']:>2}  "
              f"min_dist_min={r['min_dist_min']:.2f}Å", flush=True)
    print(f"  best total:    step {best_step_total} = {best_total:.3f}", flush=True)
    print(f"  best CDR→epi:  step {best_step_epi} = {best_cdr_to_epi:.2f}",
          flush=True)
    print(f"  best inter:    step {best_step_inter} = {best_inter:.3f}", flush=True)
    print(f"  best ipTM:     step {best_step_iptm} = {best_iptm:.3f}", flush=True)
    print(f"  best (by total) CDR: {''.join(best_seq_total[i] for i in cdr)}",
          flush=True)
    print(f"  best (by ipTM)  CDR: {''.join(best_seq_iptm[i] for i in cdr)}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Design steps between epitope re-anchors.")
    p.add_argument("--epi-threshold", type=float, default=10.0,
                   help="Å; target residues closer than this to ANY binder "
                        "residue are included in the new epitope.")
    p.add_argument("--epi-min-size", type=int, default=4)
    p.add_argument("--epi-max-size", type=int, default=40)
    p.add_argument("--epitope-cutoff", type=float, default=8.0)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v15_reanchor_snaps.json")
    p.add_argument("--init-seq", type=str, default=V2_STEP050_FULL)
    p.add_argument("--use-wt-prior", action="store_true",
                   help="Skip Full prior; use WT crystal prior (sanity check)")
    args = p.parse_args()

    setup = setup_design(epitope_cutoff=args.epitope_cutoff, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    prior_bins = prior_mask = None
    if not args.use_wt_prior:
        from design_b5_mps_v9_cacoord import predict_prior_from_full_ca
        prior_bins, prior_mask, _, _ = predict_prior_from_full_ca(
            args.init_seq, target_seq,
            num_loops=args.full_loops,
            num_sampling=args.full_samples,
            num_diffusion_samples=args.full_diffusion,
        )

    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               chunk_size=args.chunk_size,
               epi_threshold=args.epi_threshold,
               epi_min_size=args.epi_min_size,
               epi_max_size=args.epi_max_size,
               epitope_cutoff=args.epitope_cutoff,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
