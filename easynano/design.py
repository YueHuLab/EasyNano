"""V9 design loop: CA-coordinate prior + differentiable CDR optimization.

Adapted from ``cookbook/tutorials/design_target_v9.py``.

Pipeline:
  1. Predict structure prior from Full ESMFold2 (1.3B) CA coordinates
  2. Optimize CDR logits via ESMFold2-Fast (721M) distogram gradients
  3. Save snapshots for later evaluation
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import math
import json
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    ESMFOLD2_FAST, ESMFOLD2_FULL, ESMC_PATH, DEVICE, ESM_REPO,
    N_STEPS, LR, WT_LOGIT, W_EPITOPE, W_INTRA, W_INTER, W_GLOB,
    W_PRIOR, W_AA_FREQ, TEMP_MIN, LOG_EVERY, SNAPSHOT_EVERY,
    SAMPLE_STEPS, N_LOOPS, FULL_LOOPS, FULL_SAMPLING, FULL_DIFFUSION,
    N_BINS, MIN_DIST, MAX_DIST, BIN_TOLERANCE,
    TOKENS, AA_TO_TOKEN, CYS_TOK, AA_DIMS, NUM_RES_TYPES,
    AA_FREQ, MUTABLE_TOKEN,
)
from .loss import (
    compute_structure_losses, get_mid_points, build_pdb_prior,
)
from .setup import setup_target_design

sys.path.insert(0, ESM_REPO)

_CA_NAME_CODE = (ord('C') - 32, ord('A') - 32, 0, 0)


# ---- Model loading ----

def load_fast_model():
    """Load ESMFold2-Fast (721M) for the design loop."""
    print(f"Loading ESMFold2-Fast from {ESMFOLD2_FAST} ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(ESMFOLD2_FAST)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(ESMFOLD2_FAST, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded in {time.time() - t0:.1f}s, {n_params:.0f}M params", flush=True)
    return model


def load_full_model():
    """Load Full ESMFold2 (1.3B) for structure prior prediction."""
    print("Loading FULL ESMFold2 (1.3B) for prior prediction ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(ESMFOLD2_FULL)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(ESMFOLD2_FULL, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


# ---- Soft-logit machinery ----

def init_soft_logits(binder_template: str, binder_wt: str,
                     wt_logit: float = WT_LOGIT,
                     pin_cys_in_cdr: bool = True) -> torch.Tensor:
    L = len(binder_template)
    logits = torch.zeros(L, AA_DIMS)
    for i, aa in enumerate(binder_template):
        if aa != MUTABLE_TOKEN:
            idx = AA_TO_TOKEN[aa] - 2
            logits[i, :] = -10.0
            logits[i, idx] = 10.0
        else:
            wt_aa = binder_wt[i]
            wt_idx = AA_TO_TOKEN[wt_aa] - 2
            logits[i, :] = 0.5 * torch.randn(AA_DIMS)
            logits[i, wt_idx] = wt_logit
            if pin_cys_in_cdr:
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
        dtype=torch.bool, device=device)


def build_soft_res_type(soft_logits: torch.Tensor, target_one_hot: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
    binder_probs_20 = F.softmax(soft_logits / max(temperature, 1e-3), dim=-1)
    binder_probs_33 = torch.zeros(
        soft_logits.size(0), NUM_RES_TYPES,
        device=soft_logits.device, dtype=binder_probs_20.dtype)
    binder_probs_33[:, 2:22] = binder_probs_20
    binder_probs_33 = binder_probs_33.unsqueeze(0)
    return torch.cat([binder_probs_33, target_one_hot.to(binder_probs_33.device)], dim=1)


def make_target_one_hot(target_seq: str, device) -> torch.Tensor:
    L = len(target_seq)
    idx = torch.tensor([AA_TO_TOKEN[aa] for aa in target_seq], device=device).long()
    return F.one_hot(idx, num_classes=NUM_RES_TYPES).float().unsqueeze(0)


def soft_to_hard_seq(soft_logits: torch.Tensor) -> str:
    idx = soft_logits.argmax(-1).cpu().tolist()
    return "".join(TOKENS[i + 2] for i in idx)


def cdr_to_epitope_stats(disto_logits, cdr_indices, epitope_indices,
                         target_length, binder_length):
    midpoints = get_mid_points().to(disto_logits.device)
    probs = torch.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    cross = e_dist[target_length:, :target_length]
    cdr_rows = [b for b in cdr_indices]
    cdr_to_e = cross[cdr_rows][:, epitope_indices]
    return {
        "cdr_to_epitope_min": cdr_to_e.min(dim=-1).values.mean().item(),
        "cdr_to_epitope_median": cdr_to_e.min(dim=-1).values.median().item(),
        "inter_min": cross.min().item(),
        "inter_median": cross.median().item(),
    }


def aa_freq_loss(soft_logits, mutable_mask):
    freq = torch.tensor(AA_FREQ, device=soft_logits.device)
    probs = F.softmax(soft_logits, dim=-1)
    log_freq = torch.log(freq)
    expected_log = (probs * log_freq.unsqueeze(0)).sum(-1)
    nll = -expected_log * mutable_mask.float()
    return nll.sum() / (mutable_mask.sum() + 1e-8)


def reorder_bf_to_target_first(disto_bf, binder_len):
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
    return disto_bf[:, perm, :, :][:, :, perm, :]


def align_prior_to_disto(prior_bins, prior_mask, disto_target_first):
    L_p = prior_bins.size(0)
    L_d = disto_target_first.size(1)
    if L_d == L_p:
        return prior_bins, prior_mask
    if L_d < L_p:
        raise RuntimeError(f"Distogram L={L_d} < prior L={L_p}")
    return prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]


def extract_ca_per_token(coords, atom_to_token, ref_atom_name_chars, atom_mask=None):
    is_ca = (
        (ref_atom_name_chars[:, 0] == _CA_NAME_CODE[0]) &
        (ref_atom_name_chars[:, 1] == _CA_NAME_CODE[1]) &
        (ref_atom_name_chars[:, 2] == _CA_NAME_CODE[2]) &
        (ref_atom_name_chars[:, 3] == _CA_NAME_CODE[3])
    )
    if atom_mask is not None:
        is_ca = is_ca & atom_mask
    ca_atom_idx = is_ca.nonzero(as_tuple=True)[0]
    ca_tok = atom_to_token[ca_atom_idx]
    L = int(atom_to_token.max().item()) + 1
    B = coords.size(0)
    out = torch.full((B, L, 3), float("nan"), device=coords.device, dtype=coords.dtype)
    out[:, ca_tok, :] = coords[:, ca_atom_idx, :]
    return out


# ---- Structure prior prediction ----

def predict_prior_from_full_ca(binder_seq: str, target_seq: str,
                               num_loops: int = FULL_LOOPS,
                               num_sampling: int = FULL_SAMPLING,
                               num_diffusion_samples: int = FULL_DIFFUSION,
                               bin_tolerance: float = BIN_TOLERANCE,
                               n_bins: int = N_BINS,
                               min_dist: float = MIN_DIST,
                               max_dist: float = MAX_DIST):
    """Fold (binder, target) with Full ESMFold2, extract CA coords, build prior."""
    print(f"\n=== Predicting prior from CA coords (loops={num_loops}, "
          f"samples={num_sampling}, diffusion={num_diffusion_samples}) ===", flush=True)
    model = load_full_model()
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}

    binder_len = len(binder_seq)
    target_len = len(target_seq)
    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]

    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(**features, num_loops=num_loops,
                            num_sampling_steps=num_sampling,
                            num_diffusion_samples=num_diffusion_samples,
                            calculate_confidence=True)
    print(f"  Full fold took {time.time() - t0:.1f}s", flush=True)
    iptm = float(out["iptm"][0].item()) if out.get("iptm") is not None and out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if out.get("ptm") is not None and out["ptm"].numel() else None
    print(f"  predicted pTM={ptm:.3f}  ipTM={iptm:.3f}" if iptm is not None else "  confidence N/A", flush=True)

    sample_coords = out["sample_atom_coords"].float()
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    ca_coords = extract_ca_per_token(sample_coords, atom_to_token,
                                      ref_atom_name_chars, atom_mask)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_avg = ca_coords.mean(dim=0).cpu()

    # Reorder binder-first → target-first
    perm = torch.cat([torch.arange(target_len, binder_len + target_len),
                       torch.arange(0, target_len)])
    ca_avg = ca_avg[perm]
    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]

    tt_dist = ca_dist[:target_len, :target_len]
    iface_dist = ca_dist[target_len:, :target_len]
    print(f"  target-target range: [{tt_dist[tt_dist > 0].min():.2f}, "
          f"{tt_dist.max():.2f}]", flush=True)
    print(f"  interface range:     [{iface_dist[iface_dist > 0].min():.2f}, "
          f"{iface_dist.max():.2f}]", flush=True)

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len, target_length=target_len,
        target_target_dist=tt_dist, interface_dist=iface_dist,
        bin_tolerance=bin_tolerance, n_bins=n_bins,
        min_dist=min_dist, max_dist=max_dist,
    )
    print(f"  constrained pairs: {int(prior_mask.sum().item())} / {prior_mask.numel()}", flush=True)

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return prior_bins, prior_mask, iptm, ptm


# ---- Design loop ----

def run_design(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int],
    framework: str = "b5",
    steps: int = N_STEPS,
    lr: float = LR,
    seed: int = 0,
    wt_logit: float = WT_LOGIT,
    w_epitope: float = W_EPITOPE,
    w_intra: float = W_INTRA,
    w_inter: float = W_INTER,
    w_glob: float = W_GLOB,
    w_prior: float = W_PRIOR,
    w_aa_freq: float = W_AA_FREQ,
    log_every: int = LOG_EVERY,
    snapshot_every: int = SNAPSHOT_EVERY,
    sample_steps: int = SAMPLE_STEPS,
    n_loops: int = N_LOOPS,
    full_loops: int = FULL_LOOPS,
    full_samples: int = FULL_SAMPLING,
    full_diffusion: int = FULL_DIFFUSION,
    skip_prior: bool = False,
    pin_cys_in_cdr: bool = True,
    init_seq: str | None = None,
    out_dir: str = "results",
    snapshot_path: str | None = None,
    quiet: bool = False,
):
    """Run the full v9 epitope-targeted CDR design pipeline.

    Returns (history, snapshots, best_info).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup
    setup = setup_target_design(
        pdb_path=pdb_path, target_chain=target_chain,
        epitope_indices=epitope_indices, framework=framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]

    if init_seq is None:
        init_seq = binder_wt

    print(f"  Target {target_len}aa, Binder {binder_len}aa, "
          f"CDRs {len(cdr)}pos, Epitope {len(epi)}res")
    print(f"  Init CDRs: {''.join(init_seq[i] for i in cdr)}")

    # Output paths
    os.makedirs(out_dir, exist_ok=True)
    if snapshot_path is None:
        snapshot_path = os.path.join(out_dir, f"seed{seed}_snapshots.json")

    # Stage 1: Structure prior
    if skip_prior:
        prior_bins = setup["prior_bins"]
        prior_mask = setup["prior_mask"]
        print("  [skip-prior] using target-only prior (no interface)")
    else:
        prior_bins, prior_mask, prior_iptm, prior_ptm = predict_prior_from_full_ca(
            init_seq, target_seq,
            num_loops=full_loops, num_sampling=full_samples,
            num_diffusion_samples=full_diffusion,
        )

    # Stage 2: Design loop with Fast model
    model = load_fast_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    # Update global loss weights
    from . import loss as L
    L.LOSS_WEIGHTS["epitope"] = w_epitope
    L.LOSS_WEIGHTS["intra_contact"] = w_intra
    L.LOSS_WEIGHTS["inter_contact"] = w_inter
    L.LOSS_WEIGHTS["glob"] = w_glob
    L.LOSS_WEIGHTS["structure_prior"] = w_prior
    if not quiet:
        print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
              f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt,
                                    wt_logit=wt_logit, pin_cys_in_cdr=pin_cys_in_cdr).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    history = []
    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1

    init_cur = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_cur[i] for i in cdr)
    assert init_cur == init_seq, f"init seq mismatch: {init_cur} vs {init_seq}"

    if not quiet:
        print(f"\nDesigning {steps} steps (lr={lr}, wt_logit={wt_logit}, "
              f"sample={sample_steps}, loops={n_loops}) ...", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot, temperature=temperature)
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
            out = model.forward(**features, num_loops=n_loops,
                                num_sampling_steps=sample_steps,
                                num_diffusion_samples=1,
                                calculate_confidence=True)
        disto_bf = out["distogram_logits"].float()
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST,
        )
        total = losses["total_loss"] + w_aa_freq * lm

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if out.get("ptm") is not None and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if out.get("iptm") is not None and out["iptm"].numel() else None
        cdr_seq = "".join(cur_seq[i] for i in cdr)

        record = {
            "step": step, "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "inter_min": diag["inter_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "n_cdr_diff_from_init": n_cdr_diff,
        }
        history.append(record)

        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq
            best_step_total = step

        if not quiet and (step == 0 or step % log_every == 0 or step == steps):
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>8.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>7.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}  [{elapsed:>5.0f}s]", flush=True)

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            if soft_logits.grad is None:
                if not quiet:
                    print(f"  [WARN step {step}] grad is None, skipping", flush=True)
                continue
            with torch.no_grad():
                soft_logits.grad[fixed_position_mask(binder_template, DEVICE)] = 0.0
            optimizer.step()
            pin_fixed_positions(soft_logits, binder_template)

        if step > 0 and step % snapshot_every == 0:
            h1 = cdr_seq[:10] if len(cdr) > 10 else cdr_seq[:10]
            h2 = cdr_seq[10:16] if len(cdr) > 16 else ""
            h3 = cdr_seq[16:] if len(cdr) > 16 else cdr_seq[10:]
            snap = {
                "step": step, "cdr_seq": cdr_seq,
                "h1": h1, "h2": h2, "h3": h3,
                "full_seq": cur_seq,
                "inter": record["inter"],
                "epi": record["soft_epi"],
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "inter_min": record["inter_min"],
                "ptm": ptm, "iptm": iptm,
                "n_cdr_diff_from_init": n_cdr_diff,
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_cdr": init_cdr,
                    "init_full_seq": init_seq,
                    "binder_len": binder_len, "target_len": target_len,
                    "epitope": epi, "framework": framework,
                    "config": {
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra,
                        "w_inter": w_inter, "w_glob": w_glob,
                        "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps,
                        "seed": seed,
                    },
                    "snapshots": snapshots,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  Initial CDRs: {init_cdr}")
    print(f"  Final   CDRs: {cdr_seq}")
    print(f"  {len(snapshots)} snapshots saved to {snapshot_path}")

    return history, snapshots, {
        "best_seq": best_seq_total, "best_step": best_step_total,
        "best_total_loss": best_total,
        "final_cdr_seq": cdr_seq, "init_cdr": init_cdr,
    }
