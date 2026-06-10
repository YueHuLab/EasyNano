"""v10b: Iter round 3 from v9 step 48 (= v10a step 12) with OFFICIAL sampling
tricks from binder_design.py:

  - Cosine T schedule (already in v9): T_min + (1 - T_min) * 0.5*(1+cos(π t))
  - **lr scales with T**: lr = LEARNING_RATE * T  (large early, tiny late)
  - **num_sampling_steps = 1 when T >= 0.05**  (fast noisy exploration)
  - **num_sampling_steps = 50 when T < 0.05**  (final high-quality refinement)
  - **calculate_confidence = (T < 0.05)**  (save compute during exploration)
  - LR_base = 0.1 (official)

The "official" pattern is: 100+ cheap noisy steps with high effective LR
(lr = 0.1 * T, which starts at 0.1 and decays to 0.001), then a few high-
quality steps with full iptm computation for final selection.

v9 step 48 is the starting sequence (stable local optimum found in v9/v10a).
The prior is fold(v9 step 48) - same as v10a.
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
    load_model, MODEL_PATH, ESMC_PATH, DEVICE, NUM_RES_TYPES, TOKENS,
    AA_TO_TOKEN, AA_DIMS, CYS_TOK, AA_FREQ, TEMP_MIN,
    SAMPLE_STEPS_FWD, N_LOOPS_FWD, W_AA_FREQ,
    pin_fixed_positions,
)
from design_b5_mps_v9_cacoord import predict_prior_from_full_ca  # noqa: E402

# v9 step 48 full sequence (= v10a step 12 best; confirmed local optimum)
V9_STEP48_FULL = (
    "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGWYMSLGWFRQAPGQGLEAVAAI"
    "SYSGQRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVADSPQR"
    "IYKAPIRWGQGTLVTVS"
)

# Official schedule knobs
TEMP_MIN_OFFICIAL = 1e-2
T_CONFIDENCE_THRESHOLD = 0.05
FAST_SAMPLE_STEPS = 1
SLOW_SAMPLE_STEPS = 50


def run_design(steps: int = 80,
               lr_base: float = 0.1,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 4,
               snapshot_every: int = 4,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v10b_official_sched_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V9_STEP48_FULL):
    print(f"=== v10b: 3-CDR finetune with OFFICIAL sampling schedule "
          f"(steps={steps}, lr_base={lr_base}, "
          f"fast={FAST_SAMPLE_STEPS}/slow={SLOW_SAMPLE_STEPS} sampling steps) ===\n",
          flush=True)
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

    model = load_model()
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
    v10b_template = "".join(template_list)
    assert v10b_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v10b_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr_base)

    fixed_mask = fixed_position_mask(v10b_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} (H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)")

    history = []
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

    print(f"\nDesigning {steps} steps (official schedule: T cosine, "
          f"lr=T*LR, num_sampling_steps=1->50 at T<0.05) ...",
          flush=True)
    header = (f"  {'step':>4}  {'T':>5}  {'lr*':>6}  {'total':>7}  {'epi':>6}  "
              f"{'inter':>6}  {'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  "
              f"{'samp':>4}  CDR_seq")
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN_OFFICIAL + (1 - TEMP_MIN_OFFICIAL) * remaining

        # Official schedule: high-quality sampling only in the final phase
        calculate_confidence = temperature < T_CONFIDENCE_THRESHOLD
        sample_steps = SLOW_SAMPLE_STEPS if calculate_confidence else FAST_SAMPLE_STEPS
        current_lr = lr_base * temperature

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

        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
                calculate_confidence=calculate_confidence,
            )
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
        ptm = float(out["ptm"][0].item()) if (calculate_confidence and "ptm" in out
                                              and out["ptm"].numel()) else None
        iptm = float(out["iptm"][0].item()) if (calculate_confidence and "iptm" in out
                                                and out["iptm"].numel()) else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step, "temperature": float(temperature),
            "lr": float(current_lr), "sample_steps": sample_steps,
            "calculate_confidence": calculate_confidence,
            "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "n_cdr_diff_from_init": n_cdr_diff,
        }
        history.append(record)

        if step == 0 or step % log_every == 0 or step == steps or calculate_confidence:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {temperature:>5.3f}  {current_lr:>6.4f}  "
                  f"{record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {sample_steps:>4}  {cdr_seq}  "
                  f"[{elapsed:>5.0f}s]",
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
            # **Official trick**: lr scales with T (decays as temperature drops)
            for g in optimizer.param_groups:
                g["lr"] = current_lr
            if step % log_every == 0:
                print(f"  [step {step}] T={temperature:.3f} lr={current_lr:.4f} "
                      f"grad_norm={g_norm:.4f} max={soft_logits.grad.abs().max().item():.4f} "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, v10b_template)

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
                        "starting_from": "v9 step 48 (= v10a step 12)",
                        "prior_source": "Full-predicted v9 step 48 3D CA coords",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "steps": steps, "lr_base": lr_base, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops,
                        "TEMP_MIN": TEMP_MIN_OFFICIAL,
                        "T_CONFIDENCE_THRESHOLD": T_CONFIDENCE_THRESHOLD,
                        "FAST_SAMPLE_STEPS": FAST_SAMPLE_STEPS,
                        "SLOW_SAMPLE_STEPS": SLOW_SAMPLE_STEPS,
                        "lr_schedule": "lr = lr_base * T (cosine)",
                        "seed": seed,
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
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr-base", type=float, default=0.1)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v10b_official_sched_snaps.json")
    p.add_argument("--init-seq", type=str, default=V9_STEP48_FULL)
    p.add_argument("--use-wt-prior", action="store_true")
    args = p.parse_args()

    print(f"=== v10b: iter round 3 from v9 step 48 with official schedule ===",
          flush=True)
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
    run_design(steps=args.steps, lr_base=args.lr_base, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
