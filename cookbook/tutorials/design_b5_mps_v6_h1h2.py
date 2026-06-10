"""v6: Refine CDR-H1 and CDR-H2 starting from v2 step050 (the best H3 design).

Replaces the buggy v5. v5 had three bugs:
  1. custom_fixed_mask compared against 'X' instead of MUTABLE_TOKEN '#'
  2. Gradient zeroing used fixed_position_mask (which doesn't know about H3)
  3. No pin_fixed_positions call after optimizer.step()

v6 fixes all three by:
  1. Building the template so H3 is marked as a non-mutable AA letter
  2. Using v2's fixed_position_mask correctly (now also covers H3)
  3. Calling pin_fixed_positions after every optimizer.step()

Template construction:
  - Framework: v2 framework AA (same as WT; v2 didn't touch framework)
  - H1+H2: MUTABLE_TOKEN '#' (mutable)
  - H3: v2 step050's AA letter (treated as fixed)

Starting point: v2 step050's full sequence
  H1 = GLQIGYGVYM (10 AAs, init at logit=3.0 + noise)
  H2 = SYSGQS (6 AAs, init at logit=3.0 + noise)
  H3 = RVVTDSYQPIYKAPIR (16 AAs, pinned at logit=10)
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
    compute_structure_losses, get_mid_points,
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

# v2 step050 starting point (best H3 design so far)
V2_STEP050_FULL = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAI"
                   "SYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQ"
                   "PIYKAPIRWGQGTLVTVS")
V2_H3 = "RVVTDSYQPIYKAPIR"  # Pinned H3 (16 AAs)

# CDR positions (0-based, from v2's setup_design output):
H1_INDICES = list(range(25, 35))  # 10 positions
H2_INDICES = list(range(54, 60))  # 6 positions
H3_INDICES = list(range(101, 117))  # 16 positions


def build_v6_template(v2_seq: str) -> str:
    """Build v6 template: framework + H3 = letters (fixed), H1+H2 = '#' (mutable)."""
    template = list(v2_seq)
    for i in H1_INDICES + H2_INDICES:
        template[i] = MUTABLE_TOKEN
    return "".join(template)


def run_design(steps: int = 100,
               lr: float = 0.3,
               wt_logit: float = 3.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 5,
               snapshot_every: int = 5,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v6_h1h2_snaps.json"):
    print(f"=== v6: Refine CDR-H1+H2 from v2 step050 (Fast, lr={lr}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    v6_template = build_v6_template(V2_STEP050_FULL)
    n_mutable = v6_template.count(MUTABLE_TOKEN)
    assert n_mutable == 16, f"Expected 16 mutable (H1+H2), got {n_mutable}"

    target_len = len(target_seq)
    binder_len = len(v6_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]  # all 3 CDRs (32 positions)
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}")
    print(f"  H1 (init): {V2_STEP050_FULL[25:35]}  H2 (init): {V2_STEP050_FULL[54:60]}  H3 (pinned): {V2_H3}")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    # init: V2_STEP050_FULL provides AAs for all positions; template marks H1+H2 as '#'
    # → framework (95) + H3 (16) get logit=10 for v2's AA (pinned)
    # → H1+H2 (16) get noisy init with logit=3.0 at v2's H1/H2 AA
    soft_logits = init_soft_logits(v6_template, V2_STEP050_FULL, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v6_template, DEVICE)  # True for non-'#' (framework + H3)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} (H1+H2)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95 + H3=16)")

    history = []
    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_inter = float("inf")
    best_seq_inter = ""
    best_step_inter = -1
    best_iptm = -1
    best_seq_iptm = ""
    best_step_iptm = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1

    init_seq = soft_to_hard_seq(soft_logits)
    print(f"\n  init H1: {init_seq[25:35]}  H2: {init_seq[54:60]}  H3: {init_seq[101:117]}")

    # Verify init sequence matches v2 for fixed positions
    fixed_diffs = [(i, V2_STEP050_FULL[i], init_seq[i])
                   for i in range(len(init_seq))
                   if fixed_mask[i].item() and init_seq[i] != V2_STEP050_FULL[i]]
    if fixed_diffs:
        print(f"  WARNING: {len(fixed_diffs)} fixed positions differ at init", flush=True)
    else:
        print(f"  ✓ All fixed positions match v2_step050 at init", flush=True)

    print(f"\nDesigning {steps} steps ...", flush=True)
    header = (f"  {'step':>4}  {'total':>7}  {'epi':>6}  {'inter':>6}  "
              f"{'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  H1+H2_seq")
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

        # Sanity checks
        cur_h3 = cur_seq[101:117]
        if cur_h3 != V2_H3:
            print(f"  WARNING: H3 changed at step {step}: {cur_h3}", flush=True)
        cdr_set = set(H1_INDICES + H2_INDICES + H3_INDICES)
        n_fw_diff = sum(1 for i in range(len(cur_seq))
                        if i not in cdr_set and cur_seq[i] != V2_STEP050_FULL[i])
        if n_fw_diff > 0 and step % log_every == 0:
            print(f"  WARNING: framework has {n_fw_diff} diffs at step {step}", flush=True)

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

        h1h2 = cur_seq[25:35] + cur_seq[54:60]
        record = {
            "step": step, "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "h1h2": h1h2,
        }
        history.append(record)

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {h1h2}  [{elapsed:>5.0f}s]", flush=True)

        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq; best_step_total = step
        if losses["inter_contact_loss"].item() < best_inter:
            best_inter = losses["inter_contact_loss"].item()
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
                soft_logits.grad[fixed_mask] = 0.0  # zero fixed (framework + H3)
            g_norm = soft_logits.grad.norm().item()
            if step % log_every == 0:
                print(f"  [step {step}] grad_norm={g_norm:.4f}  max={soft_logits.grad.abs().max().item():.4f}", flush=True)
            optimizer.step()
            # CRITICAL: re-pin fixed positions to prevent drift
            pin_fixed_positions(soft_logits, v6_template)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "h1": cur_seq[25:35],
                "h2": cur_seq[54:60],
                "h3": cur_seq[101:117],
                "h1h2": h1h2,
                "full_seq": cur_seq,
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "ptm": ptm, "iptm": iptm,
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_full_seq": V2_STEP050_FULL,
                    "binder_len": binder_len,
                    "config": {
                        "model": "ESMFold2-Fast (721M) for design",
                        "starting_from": "v2 step050",
                        "fixed": "framework (95) + CDR-H3 (16) = RVVTDSYQPIYKAPIR",
                        "mutable": "H1 (10 AAs) + H2 (6 AAs)",
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
    print(f"  best inter:    step {best_step_inter} = {best_inter:.3f}")
    print(f"  best ipTM:     step {best_step_iptm} = {best_iptm:.3f}")
    print(f"  best (by total): H1={best_seq_total[25:35]} H2={best_seq_total[54:60]}")
    print(f"  best (by ipTM):  H1={best_seq_iptm[25:35]} H2={best_seq_iptm[54:60]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--wt-logit", type=float, default=3.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v6_h1h2_snaps.json")
    args = p.parse_args()
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, seed=args.seed,
               log_every=args.log_every, snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path)
