"""Full ESMFold2 evaluation: fold candidates and report ipTM/pTM.

Adapted from ``cookbook/tutorials/eval_target_snapshots.py`` and
``eval_all_targets.py``.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import (
    ESMFOLD2_FULL, ESMC_PATH, DEVICE, ESM_REPO,
    FULL_LOOPS, FULL_SAMPLING,
    N_BINS, MIN_DIST, MAX_DIST,
)
from .loss import get_mid_points, compute_structure_losses
from .setup import setup_target_design

sys.path.insert(0, ESM_REPO)


def load_model():
    print(f"Loading Full ESMFold2 from {ESMFOLD2_FULL} ...", flush=True)
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(ESMFOLD2_FULL)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(ESMFOLD2_FULL, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def inject(cdr_subseq: str, template: str, cdr_indices: list[int]) -> str:
    chars = list(template)
    for i, c in zip(cdr_indices, cdr_subseq):
        chars[i] = c
    return "".join(chars)


def expected_distance(disto):
    midpoints = get_mid_points().to(disto.device)
    probs = torch.softmax(disto, dim=-1)
    return (probs * midpoints).sum(-1)


def fold_one(model, binder_seq: str, target_seq: str,
             num_loops: int = FULL_LOOPS,
             num_sampling: int = FULL_SAMPLING):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    with torch.inference_mode():
        out = model.forward(**features, num_loops=num_loops,
                            num_sampling_steps=num_sampling,
                            num_diffusion_samples=1,
                            calculate_confidence=True)
    return out


def eval_one(model, binder_seq, target_seq, binder_len, target_len,
             cdr_indices, epi_indices, prior_bins, prior_mask):
    out = fold_one(model, binder_seq, target_seq)
    disto_bf = out["distogram_logits"].float()
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
    disto = disto_bf[:, perm, :, :][:, :, perm, :]
    L_p = prior_bins.size(0)
    if disto.size(1) != L_p:
        disto = disto[:, :L_p, :L_p, :]

    e_dist = expected_distance(disto)[0]
    cross = e_dist[target_len:, :target_len]
    cdr_to_e = cross[cdr_indices][:, epi_indices]
    cdr_min = cdr_to_e.min(dim=-1).values.mean().item()

    iptm = float(out["iptm"][0].item()) if out.get("iptm") is not None and out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if out.get("ptm") is not None and out["ptm"].numel() else None
    return iptm, ptm, cdr_min


def evaluate_snapshots(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int],
    framework: str = "b5",
    snapshot_path: str | None = None,
    snapshots: list[dict] | None = None,
    n_top: int = 5,
    out: str | None = None,
    include_wt: bool = True,
):
    """Evaluate design snapshots with Full ESMFold2.

    Provide either snapshot_path (JSON file from design) or snapshots (list of dicts).
    """
    model = load_model()
    setup = setup_target_design(
        pdb_path=pdb_path, target_chain=target_chain,
        epitope_indices=epitope_indices, framework=framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    cdr_idx = setup["cdr_indices"]
    epi_idx = setup["epitope_token_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    if snapshots is None and snapshot_path:
        with open(snapshot_path) as f:
            data = json.load(f)
        snapshots = data.get("snapshots", [])

    if not snapshots:
        print("No snapshots to evaluate.")
        return []

    # Deduplicate by full CDR sequence
    seen = set()
    unique = []
    for s in snapshots:
        cdr = s.get("cdr_seq", "")
        if cdr not in seen:
            seen.add(cdr)
            unique.append(s)
    print(f"  {len(unique)} unique CDR sequences from {len(snapshots)} snapshots")

    # Rank by design-time inter loss, take top-N
    ranked = sorted(unique, key=lambda s: s.get("inter", float("inf")))[:n_top]

    results = []
    if include_wt:
        wt_cdr = "".join(binder_wt[i] for i in cdr_idx)
        print(f"\n  {'Name':>16}  {'ipTM':>6}  {'pTM':>6}  {'CDR→epi':>8}  CDR_seq")
        print(f"  {'-'*70}")
        t0 = time.time()
        wt_iptm, wt_ptm, wt_cdr_epi = eval_one(
            model, binder_wt, target_seq, binder_len, target_len,
            cdr_idx, epi_idx, prior_bins, prior_mask)
        print(f"  {'WT':>16}  {wt_iptm:>6.3f}  {wt_ptm:>6.3f}  {wt_cdr_epi:>8.1f}  "
              f"{wt_cdr}  [{time.time() - t0:.0f}s]")
        results.append({"name": "WT", "iptm": wt_iptm, "ptm": wt_ptm,
                        "cdr_to_epi": wt_cdr_epi, "cdr_seq": wt_cdr})

    for snap in ranked:
        cdr_sub = snap["cdr_seq"]
        if len(cdr_sub) != len(cdr_idx):
            continue
        full = inject(cdr_sub, binder_template, cdr_idx)
        try:
            t0 = time.time()
            iptm, ptm, cdr_epi = eval_one(
                model, full, target_seq, binder_len, target_len,
                cdr_idx, epi_idx, prior_bins, prior_mask)
            dt = time.time() - t0
            results.append({
                "name": f"step{snap['step']}",
                "step": snap["step"],
                "iptm": iptm, "ptm": ptm,
                "cdr_to_epi": cdr_epi,
                "cdr_seq": cdr_sub,
                "full_seq": full,
                "time_s": dt,
            })
            print(f"  {'step'+str(snap['step']):>16}  {iptm:>6.3f}  {ptm:>6.3f}  "
                  f"{cdr_epi:>8.1f}  {cdr_sub}  [{dt:.0f}s]")
        except Exception as e:
            print(f"  [ERR] step {snap['step']}: {e}")

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out}")

    # Best
    valid = [r for r in results if r.get("iptm") is not None]
    if valid:
        best = max(valid, key=lambda r: r["iptm"])
        print(f"\n  Best: {best['name']}  ipTM={best['iptm']:.3f}  "
              f"CDR→epi={best['cdr_to_epi']:.1f}Å  {best['cdr_seq']}")

    return results
