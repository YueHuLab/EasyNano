"""Random CDR baseline: null distribution for statistical significance.

Adapted from ``cookbook/tutorials/random_baseline.py``.
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
    FULL_LOOPS, FULL_SAMPLING, AA20_NO_CYS,
)
from .setup import setup_target_design

sys.path.insert(0, ESM_REPO)


def load_model():
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


def random_cdr_seq(cdr_length: int, rng: np.random.RandomState) -> str:
    return "".join(rng.choice(list(AA20_NO_CYS), size=cdr_length))


def inject(cdr_subseq: str, template: str, cdr_indices: list[int]) -> str:
    chars = list(template)
    for i, c in zip(cdr_indices, cdr_subseq):
        chars[i] = c
    return "".join(chars)


def fold_one(model, binder_seq: str, target_seq: str,
             num_loops=FULL_LOOPS, num_sampling=FULL_SAMPLING):
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


def run_baseline(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int],
    framework: str = "b5",
    n_random: int = 30,
    seed: int = 42,
    out: str | None = None,
):
    """Generate N random CDR sequences and evaluate with Full ESMFold2."""
    rng = np.random.RandomState(seed)
    setup = setup_target_design(
        pdb_path=pdb_path, target_chain=target_chain,
        epitope_indices=epitope_indices, framework=framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    cdr_indices = setup["cdr_indices"]
    cdr_len = len(cdr_indices)
    wt_cdr = "".join(binder_wt[i] for i in cdr_indices)

    print(f"=== Random CDR baseline: {cdr_len} CDR positions, "
          f"{n_random} random sequences ===")
    print(f"  WT CDR: {wt_cdr}")

    model = load_model()
    results = []

    print(f"\n  {'Name':>16}  {'ipTM':>6}  {'pTM':>6}  CDR_seq")
    print(f"  {'-'*70}")
    t0 = time.time()
    out_fold = fold_one(model, binder_wt, target_seq)
    wt_iptm = float(out_fold["iptm"][0].item())
    wt_ptm = float(out_fold["ptm"][0].item())
    print(f"  {'WT':>16}  {wt_iptm:>6.3f}  {wt_ptm:>6.3f}  "
          f"{wt_cdr}  [{time.time() - t0:.0f}s]")
    results.append({"name": "WT", "iptm": wt_iptm, "ptm": wt_ptm,
                    "cdr_seq": wt_cdr})

    iptm_values = []
    for i in range(n_random):
        rcdr = random_cdr_seq(cdr_len, rng)
        full = inject(rcdr, binder_template, cdr_indices)
        try:
            t0 = time.time()
            out_fold = fold_one(model, full, target_seq)
            iptm = float(out_fold["iptm"][0].item())
            ptm = float(out_fold["ptm"][0].item())
            dt = time.time() - t0
            iptm_values.append(iptm)
            results.append({"name": f"random_{i + 1}", "iptm": iptm,
                            "ptm": ptm, "cdr_seq": rcdr})
            if (i + 1) % 10 == 0 or i == 0:
                median = np.median(iptm_values)
                print(f"  {'random_'+str(i+1):>16}  {iptm:>6.3f}  {ptm:>6.3f}  "
                      f"{rcdr}  [{dt:.0f}s]  [{i+1}/{n_random}, median={median:.3f}]")
        except Exception as e:
            print(f"  [ERR] random_{i+1}: {e}")

    iptm_arr = np.array(iptm_values)
    summary = {
        "target_pdb": pdb_path, "target_chain": target_chain,
        "framework": framework, "cdr_len": cdr_len,
        "n_random": n_random, "wt_cdr": wt_cdr,
        "wt_iptm": wt_iptm, "wt_ptm": wt_ptm,
        "random_iptm_mean": float(iptm_arr.mean()),
        "random_iptm_median": float(np.median(iptm_arr)),
        "random_iptm_std": float(iptm_arr.std()),
        "random_iptm_min": float(iptm_arr.min()),
        "random_iptm_max": float(iptm_arr.max()),
        "random_iptm_p90": float(np.percentile(iptm_arr, 90)),
        "results": results,
    }

    print(f"\n=== Summary ===")
    print(f"  WT             ipTM = {wt_iptm:.4f}")
    print(f"  Random (n={len(iptm_values)})  ipTM = "
          f"{summary['random_iptm_median']:.4f} ± {summary['random_iptm_std']:.4f}")
    z = (wt_iptm - summary['random_iptm_mean']) / max(summary['random_iptm_std'], 1e-8)
    print(f"  WT z-score = {z:.1f}σ")

    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to {out}")

    return summary
