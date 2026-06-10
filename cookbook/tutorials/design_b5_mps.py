"""Multi-step binder design loop on Mac MPS using local ESMFold2-Fast.

Goal: take the B5 nanobody framework + a template with ``#`` at CDRs,
and design CDRs that bind the B5 antigen.

Key design choices:
  * Local weights at /Users/huyue/esm-c-fold2/ESMFold2-Fast
  * Local ESMC weights at /Users/huyue/esm-c-fold2/ESMC-6B
  * The local model (ESMFold2Model) accepts a 3-D ``res_type`` probability
    tensor — the binder portion is soft, the target portion is hard
    one-hot. This makes the distogram differentiable in the binder logits.
  * Skips ESMC pseudoperplexity (24 GB LM is too heavy on MPS). The
    structure losses alone are sufficient to demonstrate binding.
  * 64-bin distogram / 2-22 Å matches the local trunk.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                 # cookbook/tutorials
sys.path.insert(0, "/Users/huyue/esm-c-fold2")                # esmscore wrapper

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS, MUTABLE_TOKEN,
    compute_structure_losses, get_mid_points,
)
from test_b5_pdb import setup_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33  # matches transformers.models.esmfold2.constants

# ESMFold2 standard token indices (0=pad, 1=gap, 2..21=AA in alphabetical order)
TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
# Map 1-letter AA -> token index
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]  # 6

# Design defaults (overridable via CLI)
N_STEPS = 30
LR = 0.5
TEMP_MIN = 0.1
LOG_EVERY = 1
SAMPLE_STEPS_FWD = 1
N_LOOPS_FWD = 0


def load_model():
    print(f"Loading ESMFold2-Fast from {MODEL_PATH} ...")
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(MODEL_PATH)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(MODEL_PATH, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    # Unwrap @torch.inference_mode() on model.forward so the distogram
    # is differentiable w.r.t. the soft res_type input.
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  inference_mode unwrapped: {not hasattr(model.forward, '__wrapped__')}")
    print(f"  loaded in {time.time() - t0:.1f}s, params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def init_soft_logits(binder_template: str, binder_wt: str) -> torch.Tensor:
    """[L_b, 20] logit parameter for the binder. Fixed positions are
    pinned to one-hot (logit 10 at the correct AA, -10 elsewhere) so
    they never change. Mutable positions init from the WT AA + small
    noise so the design starts "near" the natural sequence and the
    optimization is a gradual refinement (per the user's request)."""
    L = len(binder_template)
    logits = torch.zeros(L, AA_DIMS)
    for i, aa in enumerate(binder_template):
        if aa != MUTABLE_TOKEN:
            assert aa in TOKENS[2:22]
            idx = AA_TO_TOKEN[aa] - 2
            logits[i, :] = -10.0
            logits[i, idx] = 10.0
        else:
            wt_aa = binder_wt[i]
            assert wt_aa in TOKENS[2:22]
            wt_idx = AA_TO_TOKEN[wt_aa] - 2
            logits[i, :] = 0.5 * torch.randn(AA_DIMS)
            logits[i, wt_idx] = 5.0   # strong WT prior
            logits[i, CYS_TOK - 2] = -10.0
    return logits.requires_grad_(True)


def pin_fixed_positions(soft_logits: torch.Tensor, binder_template: str):
    """Re-pin fixed positions after an optimizer step."""
    with torch.no_grad():
        for i, aa in enumerate(binder_template):
            if aa != MUTABLE_TOKEN:
                idx = AA_TO_TOKEN[aa] - 2
                soft_logits[i, :] = -10.0
                soft_logits[i, idx] = 10.0


def fixed_position_mask(binder_template: str, device) -> torch.Tensor:
    """[L_b] bool: True for fixed positions."""
    return torch.tensor(
        [aa != MUTABLE_TOKEN for aa in binder_template],
        dtype=torch.bool, device=device
    )


def build_soft_res_type(soft_logits: torch.Tensor, target_one_hot: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
    """Build [1, L, 33] res_type matching build_complex_features layout
    (binder first, target second). Binder portion is soft, target is
    hard one-hot. Temperature anneals from 1 (soft) to ~0 (sharp)."""
    binder_probs_20 = F.softmax(soft_logits / max(temperature, 1e-3), dim=-1)
    binder_probs_33 = torch.zeros(
        soft_logits.size(0), NUM_RES_TYPES,
        device=soft_logits.device, dtype=binder_probs_20.dtype
    )
    binder_probs_33[:, 2:22] = binder_probs_20
    binder_probs_33 = binder_probs_33.unsqueeze(0)   # [1, L_b, 33]
    return torch.cat([binder_probs_33, target_one_hot.to(binder_probs_33.device)], dim=1)


def make_target_one_hot(target_seq: str, device) -> torch.Tensor:
    """[1, L_t, 33] hard one-hot for the target chain."""
    L = len(target_seq)
    idx = torch.tensor([AA_TO_TOKEN[aa] for aa in target_seq], device=device).long()
    oh = F.one_hot(idx, num_classes=NUM_RES_TYPES).float()
    return oh.unsqueeze(0)


def soft_to_hard_seq(soft_logits: torch.Tensor) -> str:
    idx = soft_logits.argmax(-1).cpu().tolist()
    return "".join(TOKENS[i + 2] for i in idx)


def expected_distance_from_disto(disto_logits: torch.Tensor) -> torch.Tensor:
    midpoints = get_mid_points(n_bins=64, min_dist=2.0, max_dist=22.0).to(disto_logits.device)
    probs = torch.softmax(disto_logits, dim=-1)
    return (probs * midpoints).sum(-1)


def cdr_to_epitope_stats(disto_logits: torch.Tensor, cdr_indices: list[int],
                         epitope_target_indices: list[int],
                         target_length: int, binder_length: int) -> dict:
    """distogram convention for our losses: [B, L, L, 64] with TARGET first."""
    e_dist = expected_distance_from_disto(disto_logits)[0]  # [L, L]
    cross = e_dist[target_length:, :target_length]           # binder->target
    cdr_rows = [b for b in cdr_indices]
    cdr_to_e = cross[cdr_rows][:, epitope_target_indices]
    return {
        "cdr_to_epitope_min": cdr_to_e.min(dim=-1).values.mean().item(),
        "cdr_to_epitope_median": cdr_to_e.min(dim=-1).values.median().item(),
    }


def run_design(steps: int = N_STEPS, lr: float = LR, seed: int = 0,
               log_every: int = LOG_EVERY, sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD):
    print("=== B5.pdb multi-step design on MPS ===\n")
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}")

    # Load model
    model = load_model()

    # Target one-hot (frozen)
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    # Soft logits parameter (must stay leaf — keep on same device throughout).
    # Initialize from the WT AA at mutable positions so the optimization
    # is a gradual refinement of the natural sequence.
    binder_wt = setup["binder_full_sequence"]
    soft_logits = init_soft_logits(binder_template, binder_wt).to(DEVICE)
    # Re-wrap as leaf (defensive — .to() can detach the graph)
    soft_logits = soft_logits.detach().requires_grad_(True)
    # Adam gives per-parameter adaptive scaling, which is critical when
    # different logit positions have very different gradient magnitudes
    # (e.g. fixed positions vs sparse CDR positions).
    optimizer = optim.Adam([soft_logits], lr=lr)

    history = []
    # Track best by epitope loss (the metric that actually corresponds to
    # binding) — total loss has many terms that may not improve together.
    best_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    init_seq = soft_to_hard_seq(soft_logits)
    print(f"\n  init CDRs: {''.join(init_seq[i] for i in cdr)}")

    print(f"\nDesigning for {steps} steps (lr={lr}, sample={sample_steps}, loops={n_loops}) ...")
    print(f"  {'step':>4}  {'total':>8}  {'intra':>7}  {'inter':>7}  "
          f"{'glob':>7}  {'epi':>7}  {'prior':>7}  {'CDR→epi Å':>10}  "
          f"{'pTM':>5}  {'ipTM':>5}  CDR_seq")
    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        # Build soft res_type
        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)  # [1, L, 33]

        # Start from the WT feature template (one-shot precomputation)
        if step == 0:
            from esmscore._complex import build_complex_features
            init_seq = setup["binder_full_sequence"]
            feats = build_complex_features(init_seq, target_seq)
            # Strip private keys
            features_template = {k: v for k, v in feats.items() if not k.startswith("_")}
            # Replace res_type with our soft version (3D) each step
            # We'll just override per-step; rest of the features stay
            # based on WT for now. The atom-level features (ref_pos etc)
            # are determined by sequence — but since the soft binder might
            # produce different atoms, we need to ALSO recompute them.
            # For a true backprop-through-atom-features we'd need to
            # vary the atom features with the soft AA. Simplest fix:
            # rebuild features every step from the current argmax seq.
            del features_template
            print("  (will rebuild atom features from current argmax each step)")

        # Argmax binder sequence (for atom-level feature lookup)
        cur_seq = soft_to_hard_seq(soft_logits)
        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        # Override res_type with our soft version
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
            )
        # Distogram is in (binder, target) order. Reorder to (target, binder)
        # which is what our losses expect.
        disto_bf = out["distogram_logits"].float()       # [1, L, L, 64]
        L = disto_bf.size(1)
        perm = torch.cat([torch.arange(binder_len, L),
                          torch.arange(0, binder_len)])
        disto = disto_bf[:, perm, :, :][:, :, perm, :]
        cdr_seq = "".join(cur_seq[i] for i in cdr)

        # Compute losses
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=prior_bins.to(DEVICE), prior_mask=prior_mask.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        record = {
            "step": step,
            "total": losses["total_loss"].item(),
            "intra": losses["intra_contact_loss"].item(),
            "inter": losses["inter_contact_loss"].item(),
            "glob": losses["glob_loss"].item(),
            "epi": losses["epitope_loss"].item(),
            "prior": losses["structure_prior_loss"].item(),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "cdr_to_epi_median": diag["cdr_to_epitope_median"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
        }
        history.append(record)

        if losses["epitope_loss"].item() < best_epi:
            best_epi = losses["epitope_loss"].item()
            best_seq_epi = cur_seq
            best_step_epi = step
        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq
            best_step_total = step

        if step % log_every == 0 or step == steps:
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>8.3f}  "
                  f"{record['intra']:>7.3f}  {record['inter']:>7.3f}  "
                  f"{record['glob']:>7.3f}  {record['epi']:>7.3f}  "
                  f"{record['prior']:>7.3f}  {diag['cdr_to_epitope_min']:>10.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}")

        if step == steps:
            break

        # Backprop + step
        optimizer.zero_grad()
        losses["total_loss"].backward()
        if soft_logits.grad is None:
            print(f"  [WARN step {step}] soft_logits.grad is None — "
                  f"backprop did not reach the logits. Skipping update.")
            continue
        # Mask out fixed positions in the gradient
        with torch.no_grad():
            mask = fixed_position_mask(binder_template, DEVICE)
            soft_logits.grad[mask] = 0.0
        # Diagnostic: grad magnitude on mutable positions
        g = soft_logits.grad
        g_norm = g.norm().item()
        g_max = g.abs().max().item()
        print(f"  [step {step}] grad norm={g_norm:.4f}  max={g_max:.4f}")
        optimizer.step()
        pin_fixed_positions(soft_logits, binder_template)

    print(f"\n=== Summary ===")
    print(f"  Initial seq CDRs: {''.join(init_seq[i] for i in cdr)}")
    print(f"  Final   seq CDRs: {''.join(history[-1]['seq'][i] for i in cdr)}")
    print(f"  Best (by total) : {''.join(best_seq_total[i] for i in cdr)}  (step {best_step_total}, total {best_total:.4f})")
    print(f"  Best (by epi)   : {''.join(best_seq_epi[i] for i in cdr)}  (step {best_step_epi}, epi {best_epi:.4f})")
    return history, best_seq_epi, best_step_epi


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    p.add_argument("--num-loops", type=int, default=N_LOOPS_FWD)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_design(steps=args.steps, lr=args.lr, seed=args.seed, log_every=args.log_every,
               sample_steps=args.sample_steps, n_loops=args.num_loops)
