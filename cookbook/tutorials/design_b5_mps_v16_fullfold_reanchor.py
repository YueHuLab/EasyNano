"""v16: 3-CDR finetune with ONE-TIME Full-fold epitope re-anchor.

Differences from v9 / v14 / v15:
  - v9:  fixed epitope = 21-residue list auto-detected from input PDB
  - v14: dynamic topk = at every step, top-K closest target residues
         (self-referential, loss can be 0 with no exploration)
  - v15: periodic re-anchor = every chunk_size=4 steps, recompute epi
         from the distogram of the current step (slower self-reference,
         still moves the target every chunk)
  - v16 (this file): a single Full ESMFold2 fold at ``--reanchor-step``
         extracts the predicted binder-antigen interface from the
         realized 3D CA coordinates of the *current best binder*.  That
         epi replaces the input-PDB epi for the rest of the design run.
         **No further re-anchors after the Full fold event.**

Why this should be different from v14/v15:
  - The Full fold has higher SNR than the Fast distogram, so the
    extracted epi is a stable, high-confidence target (not a noisy
    snapshot of the optimizer's current state).
  - "Don't move the epi for the rest of the run" makes the loss
    *external-to-current-state* for the second half: the optimizer
    must pull the CDRs toward a fixed target that was set from a
    different (higher-quality) source.
  - The user's diagnosis: "老动表位反复横跳" — v15 still moves the
    target too often, defeating the gradient signal.

Default config:
  - First 30 steps: standard v9 with input-PDB epi (21 residues)
  - Step 30: Full fold of current binder seq, epi = target residues
    with min CA distance < 10.0 Å to any binder residue
  - Steps 30-60: v9 with the new epi, **fixed for the rest of the run**

Pinned: framework 95 positions, mutable: 3 CDRs (32 positions), starting
from v2 step050 (same as v9), lr=0.05, steps=60.
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
from design_b5_mps_v9_cacoord import (  # noqa: E402
    load_full_model, extract_ca_per_token,
)

# v2 step050 starting point (the classic best)
V2_STEP050_FULL = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEA"
                   "VAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCA"
                   "ARVVTDSYQPIYKAPIRWGQGTLVTVS")


def fullfold_extract_epitope(
    full_model,
    binder_seq: str,
    target_seq: str,
    target_length: int,
    binder_length: int,
    threshold: float = 10.0,
    num_loops: int = 3,
    num_sampling: int = 14,
    num_diffusion_samples: int = 1,
) -> tuple[list[int], dict]:
    """Run Full ESMFold2 on (binder_seq, target_seq), extract the predicted
    interface = target residues with min CA distance < threshold of any
    binder residue.

    Returns:
      epi_list: sorted list of target token indices (0-based)
      diag: {n_within, n_target, iface_min, iface_median, full_iptm, time_s}
    """
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}

    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]

    t0 = time.time()
    with torch.inference_mode():
        out = full_model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=num_diffusion_samples,
            calculate_confidence=True,
        )
    fold_time = time.time() - t0

    sample_coords = out["sample_atom_coords"].float()  # [B*ds, n_atoms, 3]
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    if sample_coords.dim() == 2:
        sample_coords = sample_coords.unsqueeze(0)

    ca_coords = extract_ca_per_token(sample_coords, atom_to_token,
                                      ref_atom_name_chars, atom_mask)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_avg = ca_coords.mean(dim=0).cpu()  # [L, 3]

    # Reorder from binder-first to target-first
    perm = torch.cat([torch.arange(target_length, binder_length + target_length),
                       torch.arange(0, target_length)])
    ca_avg = ca_avg[perm]
    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]
    iface_dist = ca_dist[target_length:, :target_length]  # [binder, target]

    # For each target residue, min distance to any binder residue
    min_per_target = iface_dist.min(dim=0).values  # [target]
    within = (min_per_target < threshold).nonzero(as_tuple=True)[0]
    epi_list = sorted(within.tolist())
    diag = {
        "n_within": len(epi_list),
        "n_target": min_per_target.numel(),
        "iface_min": float(iface_dist[iface_dist > 0].min().item())
                      if (iface_dist > 0).any() else 0.0,
        "iface_median": float(iface_dist.median().item()),
        "min_per_target_min": float(min_per_target.min().item()),
        "min_per_target_median": float(min_per_target.median().item()),
        "full_iptm": float(out["iptm"][0].item()) if "iptm" in out
                       and out["iptm"].numel() else None,
        "full_ptm": float(out["ptm"][0].item()) if "ptm" in out
                      and out["ptm"].numel() else None,
        "time_s": fold_time,
    }
    return epi_list, diag


def run_design(steps: int = 60,
               lr: float = 0.05,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               reanchor_step: int = 30,
               full_epi_threshold: float = 10.0,
               epitope_cutoff: float = 8.0,
               log_every: int = 4,
               snapshot_every: int = 4,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               full_loops: int = 3,
               full_samples: int = 14,
               full_diffusion: int = 1,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v16_fullfold_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V2_STEP050_FULL):
    print(f"=== v16: 3-CDR finetune with ONE-TIME Full-fold re-anchor "
          f"(reanchor_step={reanchor_step}, threshold={full_epi_threshold}Å, "
          f"lr={lr}, steps={steps}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=epitope_cutoff, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]

    target_len = len(target_seq)
    binder_len = len(binder_template)
    init_epi = list(setup["epitope_token_indices"])
    cdr = setup["cdr_indices"]
    if prior_bins is None:
        prior_bins = setup["prior_bins"]
    if prior_mask is None:
        prior_mask = setup["prior_mask"]

    epi = list(init_epi)
    epi_history = [{"step": 0, "source": "input_pdb",
                    "epi": list(epi), "size": len(epi)}]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, init epitope {len(epi)}", flush=True)
    print(f"  init epitope residues: {epi}", flush=True)
    print(f"  Init: H1={init_seq[25:35]}  H2={init_seq[54:60]}  "
          f"H3={init_seq[101:117]}", flush=True)

    model = load_model()  # Fast model for design loop
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
    v16_template = "".join(template_list)
    assert v16_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v16_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v16_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} "
          f"(H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)", flush=True)

    full_model = None  # lazy load at reanchor step
    reanchor_event = None  # filled at reanchor step
    snapshots = []
    history = []
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
    assert init_cur == init_seq, f"init seq != provided init_seq"

    print(f"\nDesigning {steps} steps; Full-fold re-anchor at step "
          f"{reanchor_step} ...", flush=True)
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
            pin_fixed_positions(soft_logits, v16_template)

        # === v16 key trick: ONE Full-fold re-anchor at reanchor_step ===
        if step + 1 == reanchor_step and step < steps:
            t0 = time.time()
            if full_model is None:
                print(f"\n  [reanchor] Loading FULL ESMFold2 (1.3G) for re-anchor ...",
                      flush=True)
                full_model = load_full_model()
            print(f"  [reanchor @ step {step + 1}] Running Full fold of "
                  f"current binder ({len(cur_seq)} aa) ...", flush=True)
            new_epi, epi_diag = fullfold_extract_epitope(
                full_model, cur_seq, target_seq,
                target_length=target_len, binder_length=binder_len,
                threshold=full_epi_threshold,
                num_loops=full_loops, num_sampling=full_samples,
                num_diffusion_samples=full_diffusion,
            )
            old_set = set(epi)
            new_set = set(new_epi)
            added = sorted(new_set - old_set)
            dropped = sorted(old_set - new_set)
            epi = new_epi
            reanchor_event = {
                "step": step + 1,
                "old_size": len(old_set),
                "new_size": len(new_epi),
                "added": added,
                "dropped": dropped,
                "threshold": full_epi_threshold,
                "n_within": epi_diag["n_within"],
                "min_per_target_min": epi_diag["min_per_target_min"],
                "min_per_target_median": epi_diag["min_per_target_median"],
                "full_iptm": epi_diag["full_iptm"],
                "full_ptm": epi_diag["full_ptm"],
                "full_fold_time_s": epi_diag["time_s"],
                "binder_at_reanchor": cur_seq,
                "elapsed_total_s": time.time() - t_start,
            }
            epi_history.append({
                "step": step + 1, "source": "fullfold",
                "epi": list(epi), "size": len(epi),
                "added": added, "dropped": dropped,
                "full_iptm": epi_diag["full_iptm"],
                "full_ptm": epi_diag["full_ptm"],
            })
            print(f"  [reanchor @ step {step + 1}] DONE in "
                  f"{epi_diag['time_s']:.1f}s (Full iptm="
                  f"{epi_diag['full_iptm']:.3f} pTM="
                  f"{epi_diag['full_ptm']:.3f})", flush=True)
            print(f"    |old|={len(old_set)} |new|={len(new_epi)} "
                  f"+{len(added)} -{len(dropped)} "
                  f"within<{full_epi_threshold}Å={epi_diag['n_within']} "
                  f"min_per_target_min={epi_diag['min_per_target_min']:.2f}Å",
                  flush=True)
            print(f"    new epi residues: {epi}", flush=True)
            print(f"    total time so far: {time.time() - t0:.0f}s "
                  f"(this step including Full fold)", flush=True)

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
                        "model": "ESMFold2-Fast (721M) for design; "
                                 "ESMFold2 (1.3G) for re-anchor",
                        "starting_from": "v2 step050",
                        "prior_source": "v9 prior (Full-predicted v2 step050 3D CA coords, "
                                        "averaged over 4 diffusion samples)",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "epitope_strategy": "one-time Full-fold re-anchor at step "
                                            f"{reanchor_step}",
                        "full_epi_threshold": full_epi_threshold,
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                    "epi_history": epi_history,
                    "reanchor_event": reanchor_event,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)",
          flush=True)
    if reanchor_event is not None:
        print(f"  re-anchor @ step {reanchor_event['step']}: "
              f"|epi| {reanchor_event['old_size']}->{reanchor_event['new_size']} "
              f"(Full iptm={reanchor_event['full_iptm']:.3f}, "
              f"fold took {reanchor_event['full_fold_time_s']:.1f}s)", flush=True)
    else:
        print(f"  re-anchor did NOT fire (reanchor_step={reanchor_step} "
              f"> steps={steps})", flush=True)
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
    p.add_argument("--reanchor-step", type=int, default=30,
                   help="Design step at which to do the ONE-TIME Full-fold "
                        "re-anchor. After this step the epi is held fixed "
                        "for the remainder of the run.")
    p.add_argument("--full-epi-threshold", type=float, default=10.0,
                   help="Å; predicted interface = target residues within this "
                        "distance of any binder residue (from Full-fold CA).")
    p.add_argument("--epitope-cutoff", type=float, default=8.0)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=1)
    p.add_argument("--prior-full-loops", type=int, default=3)
    p.add_argument("--prior-full-samples", type=int, default=14)
    p.add_argument("--prior-full-diffusion", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v16_fullfold_snaps.json")
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
            num_loops=args.prior_full_loops,
            num_sampling=args.prior_full_samples,
            num_diffusion_samples=args.prior_full_diffusion,
        )

    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               reanchor_step=args.reanchor_step,
               full_epi_threshold=args.full_epi_threshold,
               epitope_cutoff=args.epitope_cutoff,
               full_loops=args.full_loops,
               full_samples=args.full_samples,
               full_diffusion=args.full_diffusion,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
