"""v14: 3-CDR finetune with DYNAMIC epitope (top-K closest target residues,
recomputed every step from the predicted distogram).

Differences from v9:
  - v9: epitope is a fixed list of 21 residues from the input PDB
        (auto-detected within 8.0 A of the binder in the starting structure)
  - v14 --epitope-mode topk: at every step, for each CDR residue, pick the
        top-K=8 closest target residues (by distogram expected distance) as
        the "current epitope". The K closest are recomputed each step.
  - v14 --epitope-mode fixed: identical to v9 (sanity check, makes the
        diff zero).

This is the "dynamic interface" test: does the fixed input-PDB epitope
constrain us from finding better binding sites? If dynamic beats fixed,
the answer is yes; if dynamic ties or is worse, the fixed epitope wasn't
the bottleneck.

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
    compute_epitope_loss, compute_topk_epitope_loss,
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


def run_design(steps: int = 60,
               lr: float = 0.05,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               epitope_mode: str = "topk",
               topk_k: int = 8,
               epitope_cutoff: float = 8.0,
               log_every: int = 4,
               snapshot_every: int = 4,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v14_dynamic_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V2_STEP050_FULL):
    print(f"=== v14: 3-CDR finetune with DYNAMIC epitope "
          f"(mode={epitope_mode}, lr={lr}, steps={steps}) ===\n", flush=True)
    assert epitope_mode in ("fixed", "topk"), f"bad epitope_mode={epitope_mode}"
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=epitope_cutoff, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]

    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    if prior_bins is None:
        prior_bins = setup["prior_bins"]
    if prior_mask is None:
        prior_mask = setup["prior_mask"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}", flush=True)
    print(f"  epitope_mode={epitope_mode}"
          + (f"  topk_k={topk_k}" if epitope_mode == "topk" else
             f"  fixed epitope (v9-style): {len(epi)} residues"), flush=True)
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
    v14_template = "".join(template_list)
    assert v14_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v14_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v14_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} "
          f"(H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)", flush=True)

    history = []
    snapshots = []
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

    print(f"\nDesigning {steps} steps ...", flush=True)
    header = (f"  {'step':>4}  {'total':>7}  {'epi':>6}  {'inter':>6}  "
              f"{'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  CDR_seq")
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

        # === Epitope loss: choose fixed or dynamic top-K ===
        if epitope_mode == "fixed":
            epi_loss = compute_epitope_loss(
                disto, binder_length=binder_len,
                epitope_token_indices=epi, cdr_indices=cdr,
                bin_distance=get_mid_points(64, 2.0, 22.0).to(disto.device),
                cutoff=epitope_cutoff,
            )
        else:  # topk — dynamic
            epi_loss = compute_topk_epitope_loss(
                disto, binder_length=binder_len,
                cdr_indices=cdr,
                bin_distance=get_mid_points(64, 2.0, 22.0).to(disto.device),
                k=topk_k,
                cutoff=epitope_cutoff,
            )

        # Build the rest of the losses (skip the embedded epitope_loss by
        # passing a placeholder, then replace it).
        # We do this by building the full losses dict and overriding epitope.
        all_losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            epitope_cutoff=epitope_cutoff,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        # Re-aggregate with the dynamic epitope loss.
        all_losses["epitope_loss"] = epi_loss
        B = disto.size(0)
        total = (LOSS_WEIGHTS["intra_contact"] * all_losses["intra_contact_loss"]
                 + LOSS_WEIGHTS["inter_contact"] * all_losses["inter_contact_loss"]
                 + LOSS_WEIGHTS["glob"] * all_losses["glob_loss"]
                 + w_epitope * epi_loss
                 + LOSS_WEIGHTS["structure_prior"] * all_losses["structure_prior_loss"])
        total = total + w_aa_freq * lm
        # Make total require grad through the right path
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
        }
        history.append(record)

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}  [{elapsed:>5.0f}s]",
                  flush=True)

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
            pin_fixed_positions(soft_logits, v14_template)

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
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_full_seq": init_seq,
                    "binder_len": binder_len,
                    "config": {
                        "model": "ESMFold2-Fast (721M) for design",
                        "starting_from": "v2 step050",
                        "prior_source": "v9 prior (Full-predicted v2 step050 3D CA coords, "
                                        "averaged over 4 diffusion samples)",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "epitope_mode": epitope_mode,
                        "topk_k": topk_k if epitope_mode == "topk" else None,
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)",
          flush=True)
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
    p.add_argument("--epitope-mode", type=str, default="topk",
                   choices=["fixed", "topk"],
                   help="fixed=v9-style 21-residue fixed epitope; "
                        "topk=dynamic per-CDR top-K closest target residues")
    p.add_argument("--topk-k", type=int, default=8,
                   help="K for topk epitope mode (per CDR residue)")
    p.add_argument("--epitope-cutoff", type=float, default=8.0)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v14_dynamic_snaps.json")
    p.add_argument("--init-seq", type=str, default=V2_STEP050_FULL)
    p.add_argument("--use-wt-prior", action="store_true",
                   help="Skip Full prior; use WT crystal prior (sanity check)")
    args = p.parse_args()

    setup = setup_design(epitope_cutoff=args.epitope_cutoff, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    prior_bins = prior_mask = None
    if not args.use_wt_prior:
        # Reuse v9's prior builder — same Full CA-coord prior as v9
        from design_b5_mps_v9_cacoord import predict_prior_from_full_ca
        prior_bins, prior_mask, _, _ = predict_prior_from_full_ca(
            args.init_seq, target_seq,
            num_loops=args.full_loops,
            num_sampling=args.full_samples,
            num_diffusion_samples=args.full_diffusion,
        )

    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               epitope_mode=args.epitope_mode,
               topk_k=args.topk_k,
               epitope_cutoff=args.epitope_cutoff,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
