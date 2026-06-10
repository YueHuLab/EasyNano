"""Generalized v9 (CA-coord prior) epitope-targeted binder design loop.

Drop-in for any target_pdb + chain + epitope, instead of hardcoded B5.

Pipeline:
  1. test_target_pdb.setup_target_design(target_pdb, target_chain, epitope, framework)
     -> target sequence, binder template, CDRs, epitope, prior
  2. Use FULL ESMFold2 (1.3G) to fold the framework's init sequence together
     with the target. Extract 3D CA coords (averaged over num_diffusion_samples)
     and rebuild the prior with interface_dist set (key v9 feature).
  3. Init soft logits (CDR positions near WT AA with noise)
  4. Adam loop on (intra, inter, glob, epitope, structure_prior) losses with
     ESMFold2-Fast (721M) for the design loop, and the CA-coord prior from (2)
     pinning the binder pose.
  5. Save CDR snapshots every SNAPSHOT_EVERY steps.

Differences from design_target.py (v2 generalized):
  - Prior is built from Full ESMFold2 3D CA coords (averaged over
    diffusion samples), with interface_dist set. v2 uses target-CA-only
    prior (no interface_dist), so v2 has no initial binder pose.
  - Loss weights: w_epitope=0.2 (vs v2's 0.05), w_prior=0.3
  - Steps=60, lr=0.05 (vs v2's 100, 0.5)

Usage:
    python design_target_v9.py \\
        --target-pdb test/5JDS.pdb --target-chain A \\
        --epitope-indices 36,38,43,45,48,50,97,98,99,101,102,103,104,105 \\
        --framework kn035 --seed 0 --steps 60
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_TF_FALLBACK"] = "1" if False else "1"
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
from test_target_pdb import setup_target_design, INIT_FRAMEWORKS  # noqa: E402

# Use Fast ESMFold2 (721M) for design — gradients are larger than Full.
MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33

TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]  # 6

AA_FREQ = torch.tensor([
    0.0743, 0.0510, 0.0443, 0.0477, 0.0290, 0.0399, 0.0604, 0.0677,
    0.0227, 0.0554, 0.0968, 0.0580, 0.0221, 0.0394, 0.0444, 0.0580,
    0.0537, 0.0127, 0.0300, 0.0660,
])

# v9 defaults (same as design_b5_mps_v9_cacoord.py)
N_STEPS = 60
LR = 0.05
TEMP_MIN = 0.1
LOG_EVERY = 4
SNAPSHOT_EVERY = 4
SAMPLE_STEPS_FWD = 5
N_LOOPS_FWD = 1
W_AA_FREQ = 0.01

# Atom-name encoding: padded to 4 chars, ord(c) - 32 (space → 0).
_CA_NAME_CODE = (ord('C') - 32, ord('A') - 32, 0, 0)


# ----- Model loading (Fast) -----
def load_model(model_path: str = MODEL_PATH):
    print(f"Loading ESMFold2 from {model_path} ...")
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(model_path)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(model_path, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s, params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


# ----- Full model loader (for prior prediction) -----
def load_full_model():
    print("Loading FULL ESMFold2 (1.3G) for prior prediction ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained("/Users/huyue/esm-c-fold2/ESMFold2")
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(
        "/Users/huyue/esm-c-fold2/ESMFold2", config=config
    ).float().to(DEVICE).eval()
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


# ----- Soft-logit machinery (v2-compatible) -----
def init_soft_logits(binder_template: str, binder_wt: str, wt_logit: float = 5.0,
                     pin_cys_in_cdr: bool = True) -> torch.Tensor:
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
        dtype=torch.bool, device=device
    )


def build_soft_res_type(soft_logits: torch.Tensor, target_one_hot: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
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
    probs = F.softmax(soft_logits, dim=-1)
    log_freq = torch.log(AA_FREQ.to(probs.device))
    expected_log = (probs * log_freq.unsqueeze(0)).sum(-1)
    nll = -expected_log * mutable_mask.float()
    return nll.sum() / (mutable_mask.sum() + 1e-8)


def reorder_bf_to_target_first(disto_bf: torch.Tensor, binder_len: int) -> torch.Tensor:
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L),
                      torch.arange(0, binder_len)])
    return disto_bf[:, perm, :, :][:, :, perm, :]


def align_prior_to_disto(prior_bins, prior_mask, disto_target_first):
    L_p = prior_bins.size(0)
    L_d = disto_target_first.size(1)
    if L_d == L_p:
        return prior_bins, prior_mask
    if L_d < L_p:
        raise RuntimeError(f"Distogram L={L_d} < prior L={L_p}; cannot pad prior.")
    return prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]


# ----- CA-coord extraction -----
def extract_ca_per_token(coords: torch.Tensor,
                          atom_to_token: torch.Tensor,
                          ref_atom_name_chars: torch.Tensor,
                          atom_mask: torch.Tensor | None = None) -> torch.Tensor:
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


# ----- v9 prior predictor -----
def predict_prior_from_full_ca(binder_seq: str, target_seq: str,
                                num_loops: int = 3,
                                num_sampling: int = 14,
                                num_diffusion_samples: int = 4,
                                bin_tolerance: float = 2.5,
                                n_bins: int = 64,
                                min_dist: float = 2.0,
                                max_dist: float = 22.0):
    """Fold (binder, target) with Full ESMFold2, extract 3D CA coords (averaged
    over diffusion samples), and build a structure prior with interface_dist set.

    Returns: (prior_bins, prior_mask, iptm, ptm)
    """
    print(f"\n=== Predicting prior from CA coords (loops={num_loops}, "
          f"samples={num_sampling}, diffusion={num_diffusion_samples}) ===",
          flush=True)
    print(f"  binder: {binder_seq[:50]}... ({len(binder_seq)} aa)", flush=True)
    print(f"  target: {target_seq[:50]}... ({len(target_seq)} aa)", flush=True)
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
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=num_diffusion_samples,
            calculate_confidence=True,
        )
    print(f"  Full fold took {time.time() - t0:.1f}s", flush=True)
    iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
    print(f"  predicted pTM={ptm:.3f}  ipTM={iptm:.3f}", flush=True)

    sample_coords = out["sample_atom_coords"].float()
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    print(f"  sample_coords shape: {tuple(sample_coords.shape)}", flush=True)

    ca_coords = extract_ca_per_token(sample_coords, atom_to_token,
                                      ref_atom_name_chars, atom_mask)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    print(f"  CA-per-token missing entries: {n_nan}", flush=True)
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_coords = ca_coords.cpu()

    ca_avg = ca_coords.mean(dim=0)
    print(f"  ca_avg shape: {tuple(ca_avg.shape)}", flush=True)

    # Reorder binder-first → target-first (matches disto / compute_structure_losses).
    perm = torch.cat([torch.arange(target_len, binder_len + target_len),
                       torch.arange(0, target_len)])
    ca_avg = ca_avg[perm]

    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]
    L = ca_dist.size(0)
    print(f"  ca_dist shape: {tuple(ca_dist.shape)}", flush=True)

    tt_dist = ca_dist[:target_len, :target_len]
    iface_dist = ca_dist[target_len:, :target_len]
    print(f"  target-target:  range [{tt_dist[tt_dist > 0].min():.2f}, "
          f"{tt_dist.max():.2f}] Å", flush=True)
    print(f"  interface:      range [{iface_dist[iface_dist > 0].min():.2f}, "
          f"{iface_dist.max():.2f}] Å", flush=True)

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len,
        target_length=target_len,
        target_target_dist=tt_dist,
        interface_dist=iface_dist,
        bin_tolerance=bin_tolerance,
        n_bins=n_bins, min_dist=min_dist, max_dist=max_dist,
    )
    print(f"  prior_bins shape: {tuple(prior_bins.shape)}", flush=True)
    print(f"  constrained pairs: {int(prior_mask.sum().item())} / "
          f"{prior_mask.numel()}", flush=True)
    print(f"  target-target constrained: "
          f"{int(prior_mask[:target_len, :target_len].sum().item())}", flush=True)
    print(f"  interface constrained:     "
          f"{int(prior_mask[target_len:, :target_len].sum().item())}", flush=True)

    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return prior_bins, prior_mask, iptm, ptm


# ----- design loop -----
def run_design(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int],
    framework: str = "b5",
    steps: int = N_STEPS,
    lr: float = LR,
    seed: int = 0,
    log_every: int = LOG_EVERY,
    snapshot_every: int = SNAPSHOT_EVERY,
    sample_steps: int = SAMPLE_STEPS_FWD,
    n_loops: int = N_LOOPS_FWD,
    w_aa_freq: float = W_AA_FREQ,
    wt_logit: float = 5.0,
    w_epitope: float = 0.2,
    w_intra: float = 0.5,
    w_inter: float = 0.5,
    w_glob: float = 0.2,
    w_prior: float = 0.3,
    snapshot_path: str = "/tmp/target_v9_snapshots.json",
    pin_cys_in_cdr: bool = True,
    init_seq: str | None = None,
    prior_bins: torch.Tensor | None = None,
    prior_mask: torch.Tensor | None = None,
    full_loops: int = 3,
    full_samples: int = 14,
    full_diffusion: int = 4,
    skip_prior: bool = False,
):
    print(f"=== v9 generalized: target {pdb_path} chain {target_chain} "
          f"framework {framework} ===")
    print(f"=== Epitope: {epitope_indices} ===\n")

    setup = setup_target_design(
        pdb_path=pdb_path,
        target_chain=target_chain,
        epitope_indices=epitope_indices,
        framework=framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    L_prior = prior_bins.size(0) if prior_bins is not None else 0

    if init_seq is None:
        init_seq = binder_wt

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}, Prior L={L_prior}")
    print(f"  Init CDRs: {''.join(init_seq[i] for i in cdr)}")

    # Predict prior with Full ESMFold2 unless provided
    if prior_bins is None and not skip_prior:
        prior_bins, prior_mask, prior_iptm, prior_ptm = predict_prior_from_full_ca(
            init_seq, target_seq,
            num_loops=full_loops,
            num_sampling=full_samples,
            num_diffusion_samples=full_diffusion,
        )
    elif skip_prior:
        # Use the target-only prior from setup (no interface)
        prior_bins = setup["prior_bins"]
        prior_mask = setup["prior_mask"]
        print("  [skip-prior] using target-only prior (no interface)")
    else:
        print("  Using provided prior")

    # Fast ESMFold2 for design loop
    model = load_model(model_path=MODEL_PATH)
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    import binder_design_hy_losses as L
    L.LOSS_WEIGHTS["epitope"] = w_epitope
    L.LOSS_WEIGHTS["intra_contact"] = w_intra
    L.LOSS_WEIGHTS["inter_contact"] = w_inter
    L.LOSS_WEIGHTS["glob"] = w_glob
    L.LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit,
                                    pin_cys_in_cdr=pin_cys_in_cdr).to(DEVICE)
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

    init_cur = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_cur[i] for i in cdr)
    assert init_cur == init_seq, f"init seq mismatch: {init_cur} vs {init_seq}"

    print(f"\nDesigning {steps} steps (lr={lr}, sample={sample_steps}, "
          f"loops={n_loops}, w_aa_freq={w_aa_freq}) ...", flush=True)
    header = (f"  {'step':>4}  {'total':>8}  {'epi':>6}  {'inter':>7}  "
              f"{'CDR→epi':>8}  {'inter_min':>9}  "
              f"{'pTM':>5}  {'ipTM':>5}  CDR_seq")
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
        non_cdr = [i for i in range(len(cur_seq)) if i not in cdr_set]
        n_fw_diff = sum(1 for i in non_cdr if cur_seq[i] != init_seq[i])
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

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step,
            "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "inter_min": diag["inter_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "n_cdr_diff_from_init": n_cdr_diff,
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

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>8.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>7.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{record['inter_min']:>9.2f}  {ptm_s:>5}  {iptm_s:>5}  "
                  f"{cdr_seq}  [{elapsed:>5.0f}s]", flush=True)

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            if soft_logits.grad is None:
                print(f"  [WARN step {step}] grad is None, skipping", flush=True)
                continue
            with torch.no_grad():
                soft_logits.grad[fixed_position_mask(binder_template, DEVICE)] = 0.0
            g_norm = soft_logits.grad.norm().item()
            if step % log_every == 0:
                print(f"  [step {step}] grad_norm={g_norm:.4f}  "
                      f"max={soft_logits.grad.abs().max().item():.4f}  "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, binder_template)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "h1": cdr_seq[:10] if len(cdr) > 10 else cdr_seq[:10],
                "h2": cdr_seq[10:16] if len(cdr) > 16 else "",
                "h3": cdr_seq[16:] if len(cdr) > 16 else cdr_seq[10:],
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
                    "binder_len": binder_len,
                    "target_len": target_len,
                    "epitope": epi,
                    "framework": framework,
                    "config": {
                        "model": "ESMFold2-Fast (721M) for design, "
                                 "Full ESMFold2 (1.3G) for prior",
                        "prior_source": "Full-predicted init seq 3D CA coords "
                                        "(averaged over diffusion samples)",
                        "fixed": "framework (positions where binder_template "
                                "is not '#')",
                        "mutable": f"CDRs (3 loops, {len(cdr)} positions)",
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  Initial CDRs: {init_cdr}")
    print(f"  Final   CDRs: {''.join(history[-1]['seq'][i] for i in cdr)}")
    print(f"  Best (by inter) CDR: {''.join(best_seq_inter[i] for i in cdr)}  "
          f"(step {best_step_inter}, inter {best_inter:.4f})")
    print(f"  Best (by epi)   CDR: {''.join(best_seq_epi[i] for i in cdr)}  "
          f"(step {best_step_epi}, epi {best_epi:.4f})")
    print(f"  Best (by total) CDR: {''.join(best_seq_total[i] for i in cdr)}  "
          f"(step {best_step_total}, total {best_total:.4f})")
    print(f"\n  {len(snapshots)} snapshots saved to {snapshot_path}")
    return history, snapshots, (best_seq_inter, best_step_inter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--epitope-indices", required=True,
                        help="Comma-separated 0-based target residue indices")
    parser.add_argument("--framework", default="b5", choices=list(INIT_FRAMEWORKS))
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    parser.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    parser.add_argument("--num-loops", type=int, default=N_LOOPS_FWD)
    parser.add_argument("--w-aa-freq", type=float, default=W_AA_FREQ)
    parser.add_argument("--wt-logit", type=float, default=5.0)
    parser.add_argument("--w-epitope", type=float, default=0.2)
    parser.add_argument("--w-intra", type=float, default=0.5)
    parser.add_argument("--w-inter", type=float, default=0.5)
    parser.add_argument("--w-glob", type=float, default=0.2)
    parser.add_argument("--w-prior", type=float, default=0.3)
    parser.add_argument("--snapshot-path", type=str,
                        default="/tmp/target_v9_snapshots.json")
    parser.add_argument("--allow-cdr-cys", action="store_true",
                        help="Don't pin Cys to -10 in CDR positions. Required "
                             "for VHH frameworks whose H3 contains a "
                             "disulfide (e.g., KN035). Default: pin.")
    parser.add_argument("--full-loops", type=int, default=3)
    parser.add_argument("--full-samples", type=int, default=14)
    parser.add_argument("--full-diffusion", type=int, default=4)
    parser.add_argument("--skip-prior", action="store_true",
                        help="Skip the Full-ESMFold2 prior prediction (use "
                             "target-only prior from setup).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epi = [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    run_design(
        pdb_path=args.target_pdb,
        target_chain=args.target_chain,
        epitope_indices=epi,
        framework=args.framework,
        steps=args.steps, lr=args.lr, seed=args.seed,
        log_every=args.log_every, snapshot_every=args.snapshot_every,
        sample_steps=args.sample_steps, n_loops=args.num_loops,
        w_aa_freq=args.w_aa_freq, wt_logit=args.wt_logit,
        w_epitope=args.w_epitope, w_intra=args.w_intra,
        w_inter=args.w_inter, w_glob=args.w_glob, w_prior=args.w_prior,
        snapshot_path=args.snapshot_path,
        pin_cys_in_cdr=not args.allow_cdr_cys,
        full_loops=args.full_loops,
        full_samples=args.full_samples,
        full_diffusion=args.full_diffusion,
        skip_prior=args.skip_prior,
    )


if __name__ == "__main__":
    main()
