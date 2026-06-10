"""Loss-only convergence analysis on the B5 antibody complex.

This script does NOT need ESMFold2 / ESMC. It treats ``distogram_logits``
as a free torch parameter (the same role it plays inside the design
loop) and asks: if we minimize our composite loss with Adam, does the
distogram converge to the PDB pose at the constrained pairs, and do
the binder CDRs approach the epitope?

If the loss-only optimization fails to recover the PDB, the full design
has no chance — the language-model branch and the structure model only
add capacity / regularization on top of this surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS,
    compute_structure_losses,
    get_mid_points,
)
from test_b5_pdb import setup_design  # noqa: E402


def expected_distance(distogram_logits: torch.Tensor) -> torch.Tensor:
    """Expectation over distance bin midpoints — what the model 'thinks' the distance is."""
    midpoints = get_mid_points(n_bins=64, min_dist=2.0, max_dist=22.0).to(distogram_logits.device)
    probs = torch.softmax(distogram_logits, dim=-1)
    return (probs * midpoints).sum(-1)


def run(steps: int = 250, lr: float = 0.5, log_every: int = 25):
    print("=== B5.pdb loss-only convergence test ===\n")
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=3.0)
    target_len = len(setup["target_sequence"])
    binder_len = len(setup["binder_template"])
    L = target_len + binder_len
    print(f"\n  L = {L} (target {target_len} + binder {binder_len})")
    print(f"  optimizing {1 * L * L * 128:,} distogram parameters\n")

    # Move tensors to CPU (no GPU here)
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    epitope = setup["epitope_token_indices"]
    cdr_indices = setup["cdr_indices"]

    # Ground-truth distance grid (for after-optimization MAE)
    midpoints = get_mid_points(n_bins=64, min_dist=2.0, max_dist=22.0)
    gt_dist = midpoints[prior_bins.clamp(min=0)]  # [L, L]

    torch.manual_seed(0)
    distogram_logits = torch.randn(1, L, L, 128, requires_grad=True) * 0.1
    distogram_logits = distogram_logits.detach().requires_grad_()
    opt = torch.optim.Adam([distogram_logits], lr=lr)

    print(f"  {'step':>5}  {'total':>8}  {'intra':>7}  {'inter':>7}  "
          f"{'glob':>7}  {'epi(Å)':>8}  {'prior':>7}  {'iface MAE Å':>11}")
    history = []
    for step in range(steps + 1):
        losses = compute_structure_losses(
            distogram_logits,
            binder_length=binder_len,
            epitope_token_indices=epitope,
            cdr_indices=cdr_indices,
            prior_bins=prior_bins,
            prior_mask=prior_mask,
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                e_dist = expected_distance(distogram_logits)[0]  # [L, L]
                iface = e_dist[target_len:, :target_len]
                gt_iface = gt_dist[target_len:, :target_len]
                iface_mask = prior_mask[target_len:, :target_len]
                if iface_mask.any():
                    iface_mae = (
                        (iface - gt_iface).abs() * iface_mask.float()
                    ).sum() / iface_mask.sum()
                else:
                    iface_mae = torch.tensor(float("nan"))
            history.append({
                "step": step,
                "total": losses["total_loss"].item(),
                "intra": losses["intra_contact_loss"].item(),
                "inter": losses["inter_contact_loss"].item(),
                "glob": losses["glob_loss"].item(),
                "epi": losses["epitope_loss"].item(),
                "prior": losses["structure_prior_loss"].item(),
                "iface_mae": iface_mae.item(),
            })
            print(f"  {step:>5}  {losses['total_loss'].item():>8.3f}  "
                  f"{losses['intra_contact_loss'].item():>7.3f}  "
                  f"{losses['inter_contact_loss'].item():>7.3f}  "
                  f"{losses['glob_loss'].item():>7.3f}  "
                  f"{losses['epitope_loss'].item():>8.3f}  "
                  f"{losses['structure_prior_loss'].item():>7.3f}  "
                  f"{iface_mae.item():>11.3f}")
        if step == steps:
            break
        opt.zero_grad()
        losses["total_loss"].backward()
        opt.step()

    # ---- Per-component analysis ----
    with torch.no_grad():
        e_dist = expected_distance(distogram_logits)[0]
        midpoints = get_mid_points(n_bins=64, min_dist=2.0, max_dist=22.0)
        # 1) Target-target reconstruction
        tt_mask = prior_mask[:target_len, :target_len]
        tt_gt = gt_dist[:target_len, :target_len]
        tt_pred = e_dist[:target_len, :target_len]
        tt_mae = ((tt_pred - tt_gt).abs() * tt_mask.float()).sum() / tt_mask.sum()
        # 2) Interface reconstruction
        if_mask = prior_mask[target_len:, :target_len]
        if_gt = gt_dist[target_len:, :target_len]
        if_pred = e_dist[target_len:, :target_len]
        if_mae = ((if_pred - if_gt).abs() * if_mask.float()).sum() / if_mask.sum()
        # 3) Per-CDR contact with the epitope (mean over CDR positions of min over epitope)
        cdr_iface = if_pred[cdr_indices]                   # [n_cdr, target_len]
        cdr_to_epi = cdr_iface[:, epitope]                 # [n_cdr, n_epi]
        min_cdr_to_epi = cdr_to_epi.min(dim=-1).values     # [n_cdr]
        # Ground-truth equivalent
        cdr_iface_gt = if_gt[cdr_indices]
        cdr_to_epi_gt = cdr_iface_gt[:, epitope]
        min_cdr_to_epi_gt = cdr_to_epi_gt.min(dim=-1).values

    print(f"\n=== After {steps} steps ===")
    print(f"  Target-target MAE   : {tt_mae.item():.3f} Å  ({int(tt_mask.sum().item())} pairs)")
    print(f"  Interface MAE       : {if_mae.item():.3f} Å  ({int(if_mask.sum().item())} pairs)")
    print(f"\n  CDR-to-epitope min-distance (Å):")
    print(f"    {'cdr_idx':>8} {'pred':>8} {'PDB':>8}  {'Δ':>7}")
    for i, idx in enumerate(cdr_indices):
        delta = min_cdr_to_epi[i].item() - min_cdr_to_epi_gt[i].item()
        print(f"    {idx:>8} {min_cdr_to_epi[i].item():>8.2f} "
              f"{min_cdr_to_epi_gt[i].item():>8.2f}  {delta:>+7.2f}")

    print(f"\n  Mean min CDR→epitope distance:")
    print(f"    predicted: {min_cdr_to_epi.mean().item():.2f} Å")
    print(f"    PDB truth: {min_cdr_to_epi_gt.mean().item():.2f} Å")
    print(f"\n  Loss weights : {LOSS_WEIGHTS}")
    return history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    run(steps=args.steps, lr=args.lr, log_every=args.log_every)
