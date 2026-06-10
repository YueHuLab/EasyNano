"""Multi-step binder design loop on Mac MPS using local ESMFold2-Fast.

Upgrades over design_b5_mps.py:
  * Uses /Users/huyue/esm-c-fold2/ESMFold2-Fast (721M, gradients flow better than Full)
  * Reports pTM/ipTM during design via calculate_confidence=True
  * Runs 100 design steps with high-quality (1-loops, 5-sample) gradient signal
  * Saves design snapshots (CDR sequences) every SNAPSHOT_EVERY steps
  * Adds a natural-AA-frequency language-model prior (cheap regularizer)
  * Tracks best by inter_contact_loss (the binding-relevant metric)
  * After design, prints a summary of all snapshots for re-evaluation

Note: The Full ESMFold2 (1.3G) model loads fine on MPS but its distogram
gradients w.r.t. the soft res_type are very small (vanishing through
the deeper trunk + extra confidence head), so the sequence doesn't
change in a 100-step design. We use the Fast model for the design
loop (proven to work in the 30-step run) and re-evaluate snapshots
with the Full model for accurate pTM/ipTM at the end.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
import json
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

# Use the Fast ESMFold2 (721M) for design — gradients are larger, ~9s/step.
# After design, the snapshots are re-evaluated with the FULL ESMFold2 (1.3G).
MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33
# Increase this if the FULL model is loaded for design instead
FULL_MODEL = False

TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]  # 6

# Natural amino acid frequency (UniProt background) — used as cheap LM prior
AA_FREQ = torch.tensor([
    0.0743,  # A
    0.0510,  # R
    0.0443,  # N
    0.0477,  # D
    0.0290,  # C
    0.0399,  # Q
    0.0604,  # E
    0.0677,  # G
    0.0227,  # H
    0.0554,  # I
    0.0968,  # L
    0.0580,  # K
    0.0221,  # M
    0.0394,  # F
    0.0444,  # P
    0.0580,  # S
    0.0537,  # T
    0.0127,  # W
    0.0300,  # Y
    0.0660,  # V
])

# Design defaults
N_STEPS = 100
LR = 0.5
TEMP_MIN = 0.1
LOG_EVERY = 5
SNAPSHOT_EVERY = 10
SAMPLE_STEPS_FWD = 5
N_LOOPS_FWD = 1
W_AA_FREQ = 0.01            # language-model frequency regularizer weight


def load_model():
    print(f"Loading FULL ESMFold2 from {MODEL_PATH} ...")
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
    # Unwrap @torch.inference_mode() for gradient flow
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s, params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


def init_soft_logits(binder_template: str, binder_wt: str, wt_logit: float = 3.0) -> torch.Tensor:
    """[L_b, 20] logit parameter for the binder. Fixed positions are pinned,
    mutable positions initialize near the WT AA with small noise.

    wt_logit: strength of the WT AA prior. 5.0 = strong prior (needs many steps to
    flip), 1.0-2.0 = easier to explore. Default 3.0 is a balance.
    """
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
            logits[i, wt_idx] = wt_logit
            logits[i, CYS_TOK - 2] = -10.0
    return logits.requires_grad_(True)


def pin_fixed_positions(soft_logits: torch.Tensor, binder_template: str):
    with torch.no_grad():
        for i, aa in enumerate(binder_template):
            if aa != MUTABLE_TOKEN:
                idx = AA_TO_TOKEN[aa] - 2
                soft_logits[i, :] = -10.0
                soft_logits[i, idx] = 10.0


def fixed_position_mask(binder_template: str, device) -> torch.Tensor:
    return torch.tensor(
        [aa != MUTABLE_TOKEN for aa in binder_template],
        dtype=torch.bool, device=device
    )


def build_soft_res_type(soft_logits: torch.Tensor, target_one_hot: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
    """Build [1, L, 33] res_type. Binder portion is soft, target is hard one-hot.
    Layout matches build_complex_features: binder first, target second."""
    binder_probs_20 = F.softmax(soft_logits / max(temperature, 1e-3), dim=-1)
    binder_probs_33 = torch.zeros(
        soft_logits.size(0), NUM_RES_TYPES,
        device=soft_logits.device, dtype=binder_probs_20.dtype
    )
    binder_probs_33[:, 2:22] = binder_probs_20
    binder_probs_33 = binder_probs_33.unsqueeze(0)
    return torch.cat([binder_probs_33, target_one_hot.to(binder_probs_33.device)], dim=1)


def make_target_one_hot(target_seq: str, device) -> torch.Tensor:
    L = len(target_seq)
    idx = torch.tensor([AA_TO_TOKEN[aa] for aa in target_seq], device=device).long()
    oh = F.one_hot(idx, num_classes=NUM_RES_TYPES).float()
    return oh.unsqueeze(0)


def soft_to_hard_seq(soft_logits: torch.Tensor) -> str:
    idx = soft_logits.argmax(-1).cpu().tolist()
    return "".join(TOKENS[i + 2] for i in idx)


def cdr_to_epitope_stats(disto_logits: torch.Tensor, cdr_indices: list[int],
                         epitope_target_indices: list[int],
                         target_length: int, binder_length: int) -> dict:
    """distogram convention for our losses: [B, L, L, 64] with TARGET first."""
    midpoints = get_mid_points().to(disto_logits.device)
    probs = torch.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    cross = e_dist[target_length:, :target_length]
    cdr_rows = [b for b in cdr_indices]
    cdr_to_e = cross[cdr_rows][:, epitope_target_indices]
    return {
        "cdr_to_epitope_min": cdr_to_e.min(dim=-1).values.mean().item(),
        "cdr_to_epitope_median": cdr_to_e.min(dim=-1).values.median().item(),
        "inter_min": cross.min().item(),
        "inter_median": cross.median().item(),
    }


def aa_freq_loss(soft_logits: torch.Tensor, mutable_mask: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under natural AA background. Penalizes rare AAs."""
    probs = F.softmax(soft_logits, dim=-1)            # [L_b, 20]
    log_freq = torch.log(AA_FREQ.to(probs.device))
    expected_log = (probs * log_freq.unsqueeze(0)).sum(-1)   # [L_b]
    nll = -expected_log * mutable_mask.float()
    return nll.sum() / (mutable_mask.sum() + 1e-8)


def reorder_bf_to_target_first(disto_bf: torch.Tensor, binder_len: int) -> torch.Tensor:
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L),
                      torch.arange(0, binder_len)])
    return disto_bf[:, perm, :, :][:, :, perm, :]


def align_prior_to_disto(prior_bins, prior_mask, disto_target_first):
    """Trim or pad the prior to match the distogram's first two dims.
    The model's distogram includes a chain-break token; the prior does not.
    For a 223-residue target + 127-residue binder, distogram is 351 (with break),
    prior is 350. We align by trimming the distogram to (L_prior, L_prior)."""
    L_p = prior_bins.size(0)
    L_d = disto_target_first.size(1)
    if L_d == L_p:
        return prior_bins, prior_mask
    if L_d < L_p:
        raise RuntimeError(f"Distogram L={L_d} < prior L={L_p}; cannot pad prior.")
    # Drop the chain-break row/col (L_p..L_d-1 should be 1 row/col, the break).
    # We slice the first L_p along both dims.
    return prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]


def run_design(steps: int = N_STEPS, lr: float = LR, seed: int = 0,
               log_every: int = LOG_EVERY, snapshot_every: int = SNAPSHOT_EVERY,
               sample_steps: int = SAMPLE_STEPS_FWD, n_loops: int = N_LOOPS_FWD,
               w_aa_freq: float = W_AA_FREQ,
               wt_logit: float = 3.0,
               w_epitope: float = 0.05,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               snapshot_path: str = "/tmp/b5_design_snapshots.json"):
    print("=== B5.pdb multi-step design on MPS (FULL ESMFold2) ===\n")
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    L_prior = prior_bins.size(0)

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}, Prior L={L_prior}")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    binder_wt = setup["binder_full_sequence"]
    # Override loss weights for this run (the design_b5_mps_v2 default is 0.05/0.5/...)
    import binder_design_hy_losses as L
    L.LOSS_WEIGHTS["epitope"] = w_epitope
    L.LOSS_WEIGHTS["intra_contact"] = w_intra
    L.LOSS_WEIGHTS["inter_contact"] = w_inter
    L.LOSS_WEIGHTS["glob"] = w_glob
    L.LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    history = []
    snapshots = []   # [(step, cdr_seq, full_seq, losses_dict)]
    best_inter = float("inf")
    best_seq_inter = ""
    best_step_inter = -1
    best_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1

    init_seq = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_seq[i] for i in cdr)
    print(f"\n  init CDRs: {init_cdr}\n")

    print(f"Designing {steps} steps (lr={lr}, sample={sample_steps}, "
          f"loops={n_loops}, w_aa_freq={w_aa_freq}) ...")
    header = (f"  {'step':>4}  {'total':>8}  {'intra':>7}  {'inter':>7}  "
              f"{'glob':>7}  {'epi':>7}  {'prior':>7}  "
              f"{'CDR→epi':>8}  {'inter_min':>9}  "
              f"{'pTM':>5}  {'ipTM':>5}  CDR_seq")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    t_start = time.time()
    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)

        cur_seq = soft_to_hard_seq(soft_logits)
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
        L_disto = disto_bf.size(1)
        # Reorder binder-first -> target-first
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        # Align prior to distogram
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        # Add the AA-frequency LM prior (only over mutable positions)
        lm = aa_freq_loss(soft_logits, mutable_mask)

        # Compute structure losses
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
        record = {
            "step": step,
            "total": float(total.item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "glob": float(losses["glob_loss"].item()),
            "epi": float(losses["epitope_loss"].item()),
            "prior": float(losses["structure_prior_loss"].item()),
            "lm": float(lm.item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "inter_min": diag["inter_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "L_disto": L_disto, "L_prior": L_prior,
        }
        history.append(record)

        if losses["inter_contact_loss"].item() < best_inter:
            best_inter = losses["inter_contact_loss"].item()
            best_seq_inter = cur_seq
            best_step_inter = step
        if losses["epitope_loss"].item() < best_epi:
            best_epi = losses["epitope_loss"].item()
            best_seq_epi = cur_seq
            best_step_epi = step
        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq
            best_step_total = step

        # Snapshot every N steps + always the final step
        if step % snapshot_every == 0 or step == steps:
            snapshots.append({
                "step": step,
                "cdr_seq": cdr_seq,
                "full_seq": cur_seq,
                "inter": record["inter"],
                "epi": record["epi"],
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "inter_min": record["inter_min"],
                "ptm": ptm, "iptm": iptm,
            })
            with open(snapshot_path, "w") as f:
                json.dump({"init_cdr": init_cdr,
                           "binder_len": binder_len,
                           "snapshots": snapshots}, f, indent=2)

        if step % log_every == 0 or step == steps:
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>8.3f}  "
                  f"{record['intra']:>7.3f}  {record['inter']:>7.3f}  "
                  f"{record['glob']:>7.3f}  {record['epi']:>7.3f}  "
                  f"{record['prior']:>7.3f}  "
                  f"{record['cdr_to_epi_min']:>8.2f}  {record['inter_min']:>9.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}")

        if step == steps:
            break

        # Backprop
        optimizer.zero_grad()
        total.backward()
        if soft_logits.grad is None:
            print(f"  [WARN step {step}] soft_logits.grad is None — "
                  f"backprop did not reach the logits. Skipping update.")
            continue
        with torch.no_grad():
            mask = fixed_position_mask(binder_template, DEVICE)
            soft_logits.grad[mask] = 0.0
        g = soft_logits.grad
        g_norm = g.norm().item()
        g_max = g.abs().max().item()
        if step % log_every == 0:
            print(f"  [step {step}] grad norm={g_norm:.4f}  max={g_max:.4f}")
        optimizer.step()
        pin_fixed_positions(soft_logits, binder_template)

    dt = time.time() - t_start
    print(f"\n=== Summary ({dt/60:.1f} min) ===")
    print(f"  Initial seq CDRs: {init_cdr}")
    print(f"  Final   seq CDRs: {''.join(history[-1]['seq'][i] for i in cdr)}")
    print(f"  Best (by inter) : {''.join(best_seq_inter[i] for i in cdr)}  "
          f"(step {best_step_inter}, inter {best_inter:.4f})")
    print(f"  Best (by epi)   : {''.join(best_seq_epi[i] for i in cdr)}  "
          f"(step {best_step_epi}, epi {best_epi:.4f})")
    print(f"  Best (by total) : {''.join(best_seq_total[i] for i in cdr)}  "
          f"(step {best_step_total}, total {best_total:.4f})")
    print(f"\n  {len(snapshots)} snapshots saved to {snapshot_path}")
    return history, snapshots, (best_seq_inter, best_step_inter)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    p.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    p.add_argument("--num-loops", type=int, default=N_LOOPS_FWD)
    p.add_argument("--w-aa-freq", type=float, default=W_AA_FREQ)
    p.add_argument("--wt-logit", type=float, default=3.0,
                   help="Strength of WT AA prior (3.0 default; lower = more exploratory)")
    p.add_argument("--w-epitope", type=float, default=0.05,
                   help="Epitope loss weight (0.05 default; 0.2-0.5 for aggressive)")
    p.add_argument("--w-intra", type=float, default=0.5)
    p.add_argument("--w-inter", type=float, default=0.5)
    p.add_argument("--w-glob", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_design_snapshots.json")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_design(steps=args.steps, lr=args.lr, seed=args.seed,
               log_every=args.log_every, snapshot_every=args.snapshot_every,
               sample_steps=args.sample_steps, n_loops=args.num_loops,
               w_aa_freq=args.w_aa_freq, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_intra=args.w_intra,
               w_inter=args.w_inter, w_glob=args.w_glob, w_prior=args.w_prior,
               snapshot_path=args.snapshot_path)
