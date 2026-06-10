"""v12: 3-CDR finetune using FULL ESMFold2 (1.3G) inside the design loop.

Direction 2 of next-steps: replace the Fast (721M) design forward pass with
the Full (1.3G) model that we use for evaluation. Hypothesis: smaller train /
eval gap → gradient directions are more aligned with the metric we actually
care about (Full iptm).

Cost: ~4x slower per step (Full is ~4x more diffusion-sample work than Fast's
single-sample truncated path). To keep wall-time manageable, we:

  - Use num_sampling_steps=10 (Fast uses ~20 in v9; Full is more expensive
    per step so we use a smaller count). Empirically Full is more accurate
    per sample, so 10 should match Fast's 20.
  - Use num_diffusion_samples=1 in the design loop (Full already does
    4-sample averaging during the prior build).
  - Reduce steps from 60 → 40 to keep total time bounded.
  - Cosine T schedule (same as v9).

Starting point: v9 step 48 (the v9/v10 basin optimum). Prior is built from
fold(v9 step 48) using Full CA-coord averaging.
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
)
from test_b5_pdb import setup_design  # noqa: E402

from design_b5_mps_v2 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    MODEL_PATH, ESMC_PATH, DEVICE, TEMP_MIN,
    W_AA_FREQ, pin_fixed_positions,
)
from design_b5_mps_v9_cacoord import (  # noqa: E402
    load_full_model, predict_prior_from_full_ca,
)

# v9 step 48 = robust local optimum
V9_STEP48_FULL = (
    "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGWYMSLGWFRQAPGQGLEAVAAI"
    "SYSGQRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVADSPQR"
    "IYKAPIRWGQGTLVTVS"
)

# Full-model design knobs
N_LOOPS_FULL = 3
SAMPLE_STEPS_FULL = 10      # Full is more expensive per step; 10 ≈ Fast's 20


def run_design(steps: int = 40,
               lr: float = 0.05,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 2,
               snapshot_every: int = 4,
               n_loops: int = N_LOOPS_FULL,
               sample_steps: int = SAMPLE_STEPS_FULL,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v12_full_in_loop_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V9_STEP48_FULL):
    print(f"=== v12: 3-CDR finetune with FULL ESMFold2 (1.3G) in design loop "
          f"(steps={steps}, lr={lr}, sample_steps={sample_steps}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
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
          f"CDRs {len(cdr)}, Epitope {len(epi)}")
    print(f"  Init: H1={init_seq[25:35]}  H2={init_seq[54:60]}  H3={init_seq[101:117]}")

    model = load_full_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    template_list = list(init_seq)
    for i in cdr:
        template_list[i] = MUTABLE_TOKEN
    v12_template = "".join(template_list)
    assert v12_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v12_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v12_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} (H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)")

    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_iptm = -1
    best_seq_iptm = ""
    best_step_iptm = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1

    init_cur = soft_to_hard_seq(soft_logits)
    print(f"\n  init CDR: {''.join(init_cur[i] for i in cdr)}")
    assert init_cur == init_seq, f"init seq != provided init_seq"

    print(f"\nDesigning {steps} steps (FULL model in loop) ...", flush=True)
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

        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        t0 = time.time()
        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
                calculate_confidence=True,
            )
        fwd_dt = time.time() - t0

        disto_bf = out["distogram_logits"].float()
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        total = losses["total_loss"] + w_aa_freq * lm

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {float(total.item()):>7.3f}  "
                  f"{float(losses['epitope_loss'].item()):>6.2f}  "
                  f"{float(losses['inter_contact_loss'].item()):>6.3f}  "
                  f"{diag['cdr_to_epitope_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}  "
                  f"[{elapsed:>5.0f}s, fwd={fwd_dt:>4.1f}s]",
                  flush=True)

        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq; best_step_total = step
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
                print(f"  [step {step}] T={temperature:.3f}  grad_norm={g_norm:.4f}  "
                      f"max={soft_logits.grad.abs().max().item():.4f}  "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, v12_template)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "h1": cdr_seq[:10],
                "h2": cdr_seq[10:16],
                "h3": cdr_seq[16:],
                "full_seq": cur_seq,
                "cdr_to_epi_min": diag["cdr_to_epitope_min"],
                "ptm": ptm, "iptm": iptm,
                "n_cdr_diff_from_init": n_cdr_diff,
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_full_seq": init_seq,
                    "binder_len": binder_len,
                    "config": {
                        "model": "ESMFold2-FULL (1.3G) for design",
                        "starting_from": "v9 step 48",
                        "prior_source": "Full-predicted v9 step 48 3D CA coords (averaged over 4 diffusion samples)",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  best total:    step {best_step_total} = {best_total:.3f}")
    print(f"  best CDR→epi:  step {best_step_epi} = {best_cdr_to_epi:.2f}")
    print(f"  best ipTM:     step {best_step_iptm} = {best_iptm:.3f}")
    print(f"  best (by total) CDR: {''.join(best_seq_total[i] for i in cdr)}")
    print(f"  best (by ipTM)  CDR: {''.join(best_seq_iptm[i] for i in cdr)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FULL,
                   help="num_sampling_steps for Full in design loop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v12_full_in_loop_snaps.json")
    p.add_argument("--init-seq", type=str, default=V9_STEP48_FULL)
    p.add_argument("--use-wt-prior", action="store_true")
    args = p.parse_args()

    print(f"=== v12: iter with FULL ESMFold2 in design loop ===", flush=True)
    print(f"  init CDR: H1={args.init_seq[25:35]}  "
          f"H2={args.init_seq[54:60]}  H3={args.init_seq[101:117]}", flush=True)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    prior_bins = prior_mask = None
    if not args.use_wt_prior:
        prior_bins, prior_mask, _, _ = predict_prior_from_full_ca(
            args.init_seq, target_seq,
            num_loops=args.full_loops,
            num_sampling=args.full_samples,
            num_diffusion_samples=args.full_diffusion,
        )
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               sample_steps=args.sample_steps,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
