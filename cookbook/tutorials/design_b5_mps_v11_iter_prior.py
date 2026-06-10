"""v11: TRUE iterative structure-conditional optimization.

The fix for v9/v10: the CA-coord prior was held FIXED during the design loop
(only updated at the start of each design round). Each step's distogram is
already the current best structure prediction, so the fixed prior was
essentially a no-op target hint.

v11 re-builds the CA-coord prior every K design steps from the Fast model's
own `sample_atom_coords` (so the prior is always aligned with the current
best structure). This is a tighter structure-conditional iteration.

Pipeline:
  1. Initialize prior from Full fold of init_seq (matches v9 quality)
  2. For each design step:
     a. Use Fast model → get distogram + sample_atom_coords
     b. structure loss with current prior
     c. backward + step
     d. If (step % K == 0): extract CA from sample_atom_coords,
        rebuild prior_bins/prior_mask from new CA-CA distances
  3. Final high-quality phase at T<0.05: num_sampling_steps=50 (like v10b)

Starts from v9 step 48 (best so far). Uses the same official schedule
(cosine T, lr = lr_base * T) as v10b.

K=4 chosen empirically: re-prior every 4 design steps balances signal
freshness vs compute cost.
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
from design_b5_mps_v9_cacoord import (  # noqa: E402
    predict_prior_from_full_ca, extract_ca_per_token,
)

# v9 step 48 full sequence (= best so far, iptm=0.661)
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

# v11: re-prior every K design steps
PRIOR_REFRESH_EVERY = 4


def rebuild_prior_from_sample_ca(out: dict, target_seq: str, binder_seq: str,
                                 features: dict, n_bins: int = 64,
                                 min_dist: float = 2.0,
                                 max_dist: float = 22.0,
                                 bin_tolerance: float = 2.5):
    """Extract CA coords from the Fast model's `out` and build a fresh prior.

    Returns: (prior_bins [L, L], prior_mask [L, L], diagnostics_dict)
    """
    sample_coords = out["sample_atom_coords"].float()  # [B*ds, n_atoms, 3]
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    if sample_coords.dim() == 2:
        sample_coords = sample_coords.unsqueeze(0)

    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features.get("atom_attention_mask", [None])[0]

    ca_coords = extract_ca_per_token(sample_coords, atom_to_token,
                                      ref_atom_name_chars, atom_mask)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_coords = ca_coords.cpu()
    # Average across batch (and diffusion samples if any) → [L, 3]
    ca_avg = ca_coords.mean(dim=0)

    target_len = len(target_seq)
    binder_len = len(binder_seq)
    # Reorder from binder-first to target-first
    perm = torch.cat([torch.arange(target_len, binder_len + target_len),
                       torch.arange(0, target_len)])
    ca_avg = ca_avg[perm]

    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]
    L = ca_dist.size(0)
    tt_dist = ca_dist[:target_len, :target_len]
    iface_dist = ca_dist[target_len:, :target_len]

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len,
        target_length=target_len,
        target_target_dist=tt_dist,
        interface_dist=iface_dist,
        bin_tolerance=bin_tolerance,
        n_bins=n_bins, min_dist=min_dist, max_dist=max_dist,
    )
    diag = {
        "interface_min": float(iface_dist[iface_dist > 0].min()),
        "interface_max": float(iface_dist.max()),
        "interface_median": float(iface_dist.median()),
        "n_constrained": int(prior_mask.sum().item()),
    }
    return prior_bins, prior_mask, diag


def run_design(steps: int = 60,
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
               prior_refresh_every: int = PRIOR_REFRESH_EVERY,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v11_iter_prior_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V9_STEP48_FULL):
    print(f"=== v11: TRUE iterative prior refresh (K={prior_refresh_every}) "
          f"+ official schedule (steps={steps}, lr_base={lr_base}) ===\n",
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
    v11_template = "".join(template_list)
    assert v11_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v11_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr_base)

    fixed_mask = fixed_position_mask(v11_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} (H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)")

    history = []
    snapshots = []
    prior_refresh_log = []
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

    print(f"\nDesigning {steps} steps with prior refresh every "
          f"{prior_refresh_every} steps ...", flush=True)
    header = (f"  {'step':>4}  {'T':>5}  {'lr*':>6}  {'total':>7}  {'prior':>5}  "
              f"{'inter':>6}  {'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  "
              f"{'samp':>4}  {'cref':>4}  CDR_seq")
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN_OFFICIAL + (1 - TEMP_MIN_OFFICIAL) * remaining

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

        if step == 0 or step % log_every == 0 or step == steps or calculate_confidence:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {temperature:>5.3f}  {current_lr:>6.4f}  "
                  f"{float(total.item()):>7.3f}  "
                  f"{float(losses['structure_prior_loss'].item()):>5.2f}  "
                  f"{float(losses['inter_contact_loss'].item()):>6.3f}  "
                  f"{diag['cdr_to_epitope_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {sample_steps:>4}  "
                  f"{len(prior_refresh_log):>4}  {cdr_seq}  [{elapsed:>5.0f}s]",
                  flush=True)

        if float(losses["total_loss"].item()) < best_total:
            best_total = float(losses["total_loss"].item())
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
            for g in optimizer.param_groups:
                g["lr"] = current_lr
            optimizer.step()
            pin_fixed_positions(soft_logits, v11_template)
            if step % log_every == 0:
                print(f"  [step {step}] T={temperature:.3f} lr={current_lr:.4f} "
                      f"grad_norm={g_norm:.4f} max={soft_logits.grad.abs().max().item():.4f} "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)

        # **v11 key trick**: refresh CA-coord prior every K steps
        if (step + 1) % prior_refresh_every == 0 and step < steps:
            try:
                t0 = time.time()
                new_pb, new_pm, pdiag = rebuild_prior_from_sample_ca(
                    out, target_seq, cur_seq, features
                )
                prior_bins = new_pb
                prior_mask = new_pm
                prior_refresh_log.append({
                    "step": step + 1,
                    "interface_min": pdiag["interface_min"],
                    "interface_max": pdiag["interface_max"],
                    "interface_median": pdiag["interface_median"],
                    "n_constrained": pdiag["n_constrained"],
                    "time_s": time.time() - t0,
                })
                if step % log_every == 0 or step < 8:
                    print(f"  [prior refresh @ step {step + 1}] "
                          f"interface_min={pdiag['interface_min']:.2f}Å "
                          f"median={pdiag['interface_median']:.2f}Å "
                          f"n_constrained={pdiag['n_constrained']} "
                          f"[{time.time() - t0:.1f}s]", flush=True)
            except Exception as e:
                print(f"  [WARN step {step}] prior refresh failed: {e}",
                      flush=True)

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
                        "model": "ESMFold2-Fast (721M) for design",
                        "starting_from": "v9 step 48 (= best so far, iptm=0.661)",
                        "prior_source": ("Full fold for init; Fast sample_atom_coords "
                                          f"refreshed every {prior_refresh_every} steps"),
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
                        "PRIOR_REFRESH_EVERY": prior_refresh_every,
                        "lr_schedule": "lr = lr_base * T (cosine)",
                        "seed": seed,
                    },
                    "snapshots": snapshots,
                    "prior_refresh_log": prior_refresh_log,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  prior refreshes: {len(prior_refresh_log)}")
    for pr in prior_refresh_log:
        print(f"    step {pr['step']:>3}  interface_min={pr['interface_min']:.2f}  "
              f"median={pr['interface_median']:.2f}  "
              f"n_constrained={pr['n_constrained']}  [{pr['time_s']:.1f}s]")
    print(f"  best total:    step {best_step_total} = {best_total:.3f}")
    print(f"  best CDR→epi:  step {best_step_epi} = {best_cdr_to_epi:.2f}")
    print(f"  best ipTM:     step {best_step_iptm} = {best_iptm:.3f}")
    print(f"  best (by total) CDR: {''.join(best_seq_total[i] for i in cdr)}")
    print(f"  best (by ipTM)  CDR: {''.join(best_seq_iptm[i] for i in cdr)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr-base", type=float, default=0.1)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--prior-refresh-every", type=int, default=PRIOR_REFRESH_EVERY)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v11_iter_prior_snaps.json")
    p.add_argument("--init-seq", type=str, default=V9_STEP48_FULL)
    p.add_argument("--use-wt-prior", action="store_true")
    args = p.parse_args()

    print(f"=== v11: TRUE iterative prior refresh from v9 step 48 ===",
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
               prior_refresh_every=args.prior_refresh_every,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
