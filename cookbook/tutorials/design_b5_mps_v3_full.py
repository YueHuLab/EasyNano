"""Full-model design loop on Mac MPS — push ipTM > 0.6 with ESMFold2 1.3G.

Key insight from probe_full_grad.py:
  - Full ESMFold2 gradients ARE usable — just need lr ~100× larger than Fast
  - Default config (3 loops, MSA on) is best — don't disable anything
  - One step at lr=50 changes 14/32 CDR residues (vs 0 at lr=0.5)

Workflow:
  1. Use Full ESMFold2 for the design loop (3-loops, 5-sample, num_diffusion_samples=1)
  2. lr=50 (Adam), 60 steps
  3. Snapshot every 10 steps
  4. Loss: same as v2 (epitope 0.05, intra 0.5, inter 0.5, glob 0.2, prior 0.3, aa_freq 0.01)
  5. Re-eval snapshots at 3-loops/14-sample (already in /tmp/b5_eval_results.json for v2)

Expected runtime: ~30 min for 60 steps (29s/step on M3 Ultra).
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

# Use the Full ESMFold2 (1.3G) for design
MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33

TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]  # 6

# Natural amino acid frequency (UniProt background) — cheap LM prior
AA_FREQ = torch.tensor([
    0.0743, 0.0510, 0.0443, 0.0477, 0.0290, 0.0399, 0.0604, 0.0677,
    0.0227, 0.0554, 0.0968, 0.0580, 0.0221, 0.0394, 0.0444, 0.0580,
    0.0537, 0.0127, 0.0300, 0.0660,
])

TEMP_MIN = 0.1
WT_LOGIT_DEFAULT = 3.0
LOG_EVERY_DEFAULT = 2
SNAPSHOT_EVERY_DEFAULT = 5
SAMPLE_STEPS_FWD = 5
N_LOOPS_FWD = 3           # Full model best config
W_AA_FREQ = 0.01


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


def init_soft_logits(binder_template, binder_wt, wt_logit=WT_LOGIT_DEFAULT):
    L = len(binder_template)
    # Initialize at 0
    logits = torch.zeros(L, AA_DIMS)
    for i, (a_t, a_w) in enumerate(zip(binder_template, binder_wt)):
        if a_t != "X":
            # Fixed position — pin with very high logit at WT AA
            idx = AA_TO_TOKEN[a_w] - 2  # token index to AA index (0-19)
            if 0 <= idx < AA_DIMS:
                logits[i, idx] = 10.0
        else:
            # Mutable — high logit at WT AA + small noise
            idx = AA_TO_TOKEN[a_w] - 2
            if 0 <= idx < AA_DIMS:
                logits[i, idx] = wt_logit
            logits[i] += 0.5 * torch.randn(AA_DIMS)
            logits[i, idx] = wt_logit  # re-pin after noise
    return logits


def build_soft_res_type(soft_logits, target_one_hot, temperature=0.5):
    # soft_logits: [L_b, 20]
    # target_one_hot: [L_t, 33]  (already one-hot in ESMFold2 res_type space)
    binder_probs = torch.softmax(soft_logits / temperature, dim=-1)  # [L_b, 20]
    # Convert 20-AA to 33-token space: tokens 0-1 are special, tokens 2-21 are AAs
    # For each AA index 0-19, the res_type token is 2+idx
    L_b = binder_probs.size(0)
    binder_33 = torch.zeros(L_b, NUM_RES_TYPES, device=binder_probs.device, dtype=binder_probs.dtype)
    binder_33[:, 2:22] = binder_probs
    # Concat binder first, then target
    return torch.cat([binder_33, target_one_hot], dim=0)


def soft_to_hard_seq(soft_logits):
    idxs = soft_logits.argmax(dim=-1).cpu().tolist()
    return "".join(TOKENS[i + 2] for i in idxs)


def make_target_one_hot(target_seq, device):
    L = len(target_seq)
    oh = torch.zeros(L, NUM_RES_TYPES, dtype=torch.float32)
    for i, aa in enumerate(target_seq):
        if aa in AA_TO_TOKEN:
            oh[i, AA_TO_TOKEN[aa]] = 1.0
    return oh.to(device)


def aa_freq_loss(soft_logits, mutable_mask):
    """Negative log-likelihood of soft distribution under natural-AA background."""
    probs = torch.softmax(soft_logits, dim=-1)  # [L, 20]
    bg = AA_FREQ.to(probs.device).to(probs.dtype)
    bg = bg / bg.sum()
    nll = -(bg.unsqueeze(0) * torch.log(probs + 1e-9)).sum(dim=-1)  # [L]
    return (nll * mutable_mask.float()).mean()


def fixed_position_mask(binder_template, device):
    """[L] bool — True for positions that should NOT be designed (fixed)."""
    L = len(binder_template)
    m = torch.zeros(L, dtype=torch.bool, device=device)
    for i, c in enumerate(binder_template):
        if c != "X":
            m[i] = True
    return m


def reorder_bf_to_target_first(disto_bf, binder_len):
    """distogram is [1, L_total, L_total, n_bins] in binder-first order.
    Reorder to target-first then binder."""
    L_total = disto_bf.size(1)
    target_len = L_total - binder_len
    # Build index mapping: binder [0, binder_len), target [binder_len, L_total)
    # New order: target first, then binder
    perm = list(range(binder_len, L_total)) + list(range(binder_len))
    perm_t = torch.tensor(perm, device=disto_bf.device, dtype=torch.long)
    # Reorder rows and cols
    d = disto_bf[0, perm_t, :, :][:, perm_t, :]  # [L_total, L_total, n_bins]
    return d.unsqueeze(0)


def align_prior_to_disto(prior_bins, prior_mask, disto):
    """prior_bins is [Lp, Lp] (binder + target sequence). disto is [1, L, L, n_bins]."""
    Lp = prior_bins.size(0)
    L = disto.size(1)
    if Lp != L:
        # Trim or pad to L
        if Lp > L:
            return prior_bins[:L, :L].to(disto.device), prior_mask[:L, :L].to(disto.device)
        else:
            pad_b = L - Lp
            pb = torch.zeros(L, L, dtype=prior_bins.dtype, device=prior_bins.device)
            pm = torch.zeros(L, L, dtype=prior_mask.dtype, device=prior_mask.device)
            pb[:Lp, :Lp] = prior_bins
            pm[:Lp, :Lp] = prior_mask
            return pb.to(disto.device), pm.to(disto.device)
    return prior_bins.to(disto.device), prior_mask.to(disto.device)


def cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len):
    """Return min distance from any CDR to any epitope residue (in Å)."""
    probs = torch.softmax(disto[0], dim=-1)  # [L, L, 64]
    mids = get_mid_points(64, 2.0, 22.0).to(disto.device)  # [64]
    expected_dist = (probs * mids.view(1, 1, -1)).sum(dim=-1)  # [L, L]
    # disto is target-first then binder. Find CDR positions in target-first layout.
    # cdr is indices into binder (0..binder_len-1). After reorder: cdr positions = target_len + cdr.
    cdr_in_tfb = [target_len + i for i in cdr]
    epi_in_tfb = epi  # epi indices are into target (0..target_len-1)
    sub = expected_dist[torch.tensor(epi_in_tfb, device=disto.device)[:, None],
                         torch.tensor(cdr_in_tfb, device=disto.device)[None, :]]
    cdr_to_epi_min = sub.min().item()
    return {"cdr_to_epitope_min": cdr_to_epi_min, "inter_min": sub.min().item()}


def run_design(steps: int = 60,
               lr: float = 50.0,
               wt_logit: float = WT_LOGIT_DEFAULT,
               w_epitope: float = 0.05,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = LOG_EVERY_DEFAULT,
               snapshot_every: int = SNAPSHOT_EVERY_DEFAULT,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_design_v3_full_snaps.json"):
    print(f"=== B5.pdb multi-step design on MPS (FULL ESMFold2, lr={lr}, "
          f"n_loops={n_loops}, sample={sample_steps}) ===\n")
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
    L_prior = prior_bins.size(0)
    binder_wt = setup["binder_full_sequence"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}, Prior L={L_prior}")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    # Override loss weights
    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    history = []
    snapshots = []
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
        # Add batch dim — Full model expects 3D [B, L, 33] for one-hot res_type
        # (2D would be interpreted as class indices by the model)
        if res_type_soft.dim() == 2:
            res_type_soft = res_type_soft.unsqueeze(0)

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
        }
        history.append(record)
        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            print(f"  {step:>4}  {record['total']:>8.4f}  {record['intra']:>7.3f}  "
                  f"{record['inter']:>7.3f}  {record['glob']:>7.3f}  {record['epi']:>7.3f}  "
                  f"{record['prior']:>7.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{record['inter_min']:>9.2f}  "
                  f"{ptm:>5.3f}  {iptm:>5.3f}  {cdr_seq[:32]}  "
                  f"[{elapsed:>5.0f}s]")

        if record["inter"] < best_inter:
            best_inter = record["inter"]; best_seq_inter = cur_seq; best_step_inter = step
        if record["epi"] < best_epi:
            best_epi = record["epi"]; best_seq_epi = cur_seq; best_step_epi = step
        if record["total"] < best_total:
            best_total = record["total"]; best_seq_total = cur_seq; best_step_total = step

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            # Gradient clipping (helps with Adam stability on Full model)
            gnorm = soft_logits.grad.norm().item()
            if gnorm > 1.0:
                soft_logits.grad.mul_(1.0 / gnorm)
            optimizer.step()
            # Renormalize to prevent logit explosion
            with torch.no_grad():
                soft_logits.clamp_(-10.0, 10.0)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "full_seq": cur_seq,
                "inter": record["inter"],
                "epi": record["epi"],
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "inter_min": record["inter_min"],
                "ptm": ptm,
                "iptm": iptm,
            }
            snapshots.append(snap)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  best inter: step {best_step_inter} = {best_inter:.4f}  seq={best_seq_inter[:40]}")
    print(f"  best epi:   step {best_step_epi} = {best_epi:.4f}  seq={best_seq_epi[:40]}")
    print(f"  best total: step {best_step_total} = {best_total:.4f}  seq={best_seq_total[:40]}")

    # Save
    out = {
        "init_cdr": init_cdr,
        "binder_len": binder_len,
        "config": {
            "model": "ESMFold2-Full (1.3G)",
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
    print(f"  init_cdr: {init_cdr}")
    print(f"  {len(snapshots)} snapshots")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=50.0)
    p.add_argument("--wt-logit", type=float, default=WT_LOGIT_DEFAULT)
    p.add_argument("--w-epitope", type=float, default=0.05)
    p.add_argument("--n-loops", type=int, default=N_LOOPS_FWD)
    p.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY_DEFAULT)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_design_v3_full_snaps.json")
    args = p.parse_args()
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, n_loops=args.n_loops,
               sample_steps=args.sample_steps, seed=args.seed,
               log_every=args.log_every, snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path)
