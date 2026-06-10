"""Refine CDR-H1 and CDR-H2 starting from v2 step050.

Strategy:
  - Start from v2 step050's full sequence (ipTM=0.538, the best so far)
  - Pin CDR-H3 at v2's value (RVVTDSYQPIYKAPIR) — don't change it
  - Free CDR-H1 (10 AAs) and CDR-H2 (6 AAs) for further optimization
  - Use Fast model + higher LR (1.0) and higher w_epitope (0.2)
  - Goal: find H1/H2 that better support the H3 contact pose

Hypothesis: v2's H3 is near its geometric limit (~10 Å), but H1/H2 can still
rearrange to better position the framework so H3 sits MORE PRECISELY on the
epitope. The design can also adjust H1/H2 to make additional contacts.
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
)

# v2 step050 starting point
V2_STEP050_CDR = "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR"
V2_STEP050_FULL = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAI"
                   "SYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQ"
                   "PIYKAPIRWGQGTLVTVS")


def init_soft_logits_from_seq(seq: str, template: str, wt_logit: float = 5.0,
                              h3_only: bool = False) -> torch.Tensor:
    """Initialize soft logits from a specific full sequence.

    - Positions in template marked 'X' (mutable): initialize at logit=wt_logit
      at the corresponding AA in seq, with small noise.
    - Positions NOT in template (fixed): pin at logit=10 at the AA in seq.
    - If h3_only: CDR-H1 and CDR-H2 also get init at logit=0 (random AA, soft).
    """
    L = len(seq)
    logits = torch.zeros(L, AA_DIMS)
    for i, (a_t, a_s) in enumerate(zip(template, seq)):
        if a_t != "X":
            # Fixed position
            idx = AA_TO_TOKEN[a_s] - 2
            if 0 <= idx < AA_DIMS:
                logits[i, idx] = 10.0
        else:
            # Mutable position (CDR)
            idx = AA_TO_TOKEN[a_s] - 2
            if 0 <= idx < AA_DIMS:
                # H3 stays pinned at WT (we want to keep v2's H3)
                if h3_only and i >= 101:
                    logits[i, idx] = 10.0  # pin H3 hard
                else:
                    logits[i, idx] = wt_logit  # H1/H2: keep at WT seed
            logits[i] += 0.3 * torch.randn(AA_DIMS)
            # Re-pin after noise
            if h3_only and i >= 101:
                logits[i, idx] = 10.0
            else:
                logits[i, idx] = wt_logit
    return logits


def custom_fixed_mask(template: str, h3_only: bool, device) -> torch.Tensor:
    """Returns True for positions that should NOT be designed.

    - All framework positions: True (fixed)
    - CDR-H3: True (fixed, keep v2's H3)
    - CDR-H1, CDR-H2: False (free to design)
    """
    L = len(template)
    m = torch.zeros(L, dtype=torch.bool, device=device)
    for i, c in enumerate(template):
        if c != "X":
            m[i] = True   # framework: fixed
        elif h3_only and i >= 101:
            m[i] = True   # CDR-H3: fixed
        else:
            m[i] = False  # CDR-H1/H2: free
    return m


def run_design(steps: int = 80,
               lr: float = 1.0,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 2,
               snapshot_every: int = 5,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_design_v5_h1h2_snaps.json"):
    print(f"=== Refine CDR-H1+H2 from v2 step050 (Fast model, lr={lr}) ===\n",
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
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    binder_wt = setup["binder_full_sequence"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}")
    print(f"  Starting from v2 step050: {V2_STEP050_CDR}")
    print(f"  Strategy: fix CDR-H3 (v2's H3 is at geometric limit), "
          f"free CDR-H1 and CDR-H2 (16 AAs)")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    # Initialize from v2 step050 sequence
    soft_logits = init_soft_logits_from_seq(
        V2_STEP050_FULL, binder_template, wt_logit=wt_logit, h3_only=True
    ).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~custom_fixed_mask(binder_template, h3_only=True, device=DEVICE)

    print(f"  mutable positions: {int(mutable_mask.sum().item())} (H1=10, H2=6)")

    history = []
    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1
    best_iptm = -1
    best_seq_iptm = ""
    best_step_iptm = -1

    init_seq = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_seq[i] for i in cdr)
    print(f"\n  init CDRs: {init_cdr}\n", flush=True)
    print(f"  init H1: {init_cdr[:10]}  H2: {init_cdr[10:16]}  H3 (FIXED): {init_cdr[16:]}")

    print(f"Designing {steps} steps (lr={lr}, sample={sample_steps}, loops={n_loops}) ...",
          flush=True)
    header = (f"  {'step':>4}  {'total':>7}  {'soft_e':>6}  {'intra':>6}  {'inter':>6}  "
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
        if res_type_soft.dim() == 2:
            res_type_soft = res_type_soft.unsqueeze(0)

        cur_seq = soft_to_hard_seq(soft_logits)
        # Sanity check: H3 should not have changed
        cur_h3 = "".join(cur_seq[i] for i in cdr[16:])
        if cur_h3 != "RVVTDSYQPIYKAPIR":
            print(f"  WARNING: H3 changed at step {step}: {cur_h3}", flush=True)

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

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        h1h2 = cdr_seq[:16]
        record = {
            "step": step,
            "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "glob": float(losses["glob_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq, "h1h2": h1h2,
        }
        history.append(record)
        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            print(f"  {step:>4}  {record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['intra']:>6.2f}  {record['inter']:>6.3f}  "
                  f"{record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm:>5.3f}  {iptm:>5.3f}  {h1h2}  "
                  f"[{elapsed:>5.0f}s]", flush=True)

        if record["total"] < best_total:
            best_total = record["total"]; best_seq_total = cur_seq; best_step_total = step
        if diag["cdr_to_epitope_min"] < best_cdr_to_epi:
            best_cdr_to_epi = diag["cdr_to_epitope_min"]
            best_seq_epi = cur_seq; best_step_epi = step
        if iptm is not None and iptm > best_iptm:
            best_iptm = iptm; best_seq_iptm = cur_seq; best_step_iptm = step

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            gnorm = soft_logits.grad.norm().item()
            if gnorm > 1.0:
                soft_logits.grad.mul_(1.0 / gnorm)
            optimizer.step()
            with torch.no_grad():
                soft_logits.clamp_(-10.0, 10.0)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "h1h2": h1h2,
                "h3": cdr_seq[16:],
                "full_seq": cur_seq,
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "ptm": ptm,
                "iptm": iptm,
            }
            snapshots.append(snap)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  best total:   step {best_step_total} = {best_total:.3f}  seq={best_seq_total[:40]}")
    print(f"  best CDR→epi: step {best_step_epi} = {best_cdr_to_epi:.2f}  seq={best_seq_epi[:40]}")
    print(f"  best ipTM:    step {best_step_iptm} = {best_iptm:.3f}  seq={best_seq_iptm[:40]}")

    out = {
        "init_cdr": init_cdr,
        "init_full_seq": init_seq,
        "binder_len": binder_len,
        "config": {
            "model": "ESMFold2-Fast (721M) for design",
            "starting_from": "v2 step050",
            "fixed_CDR_H3": "RVVTDSYQPIYKAPIR",
            "mutable_CDRs": "H1 (10 AAs) + H2 (6 AAs)",
            "steps": steps, "lr": lr, "wt_logit": wt_logit,
            "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
            "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
            "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
        },
        "snapshots": snapshots,
    }
    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSnapshots saved to {snapshot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_design_v5_h1h2_snaps.json")
    args = p.parse_args()
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, seed=args.seed,
               log_every=args.log_every, snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path)
