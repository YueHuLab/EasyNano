"""Random CDR baseline: what ipTM do random CDR sequences get?

For each target, generates N random CDR sequences (same lengths as WT,
natural AA frequencies), folds with Full ESMFold2, and reports ipTM distribution.
This establishes the null hypothesis for the design method.

Usage:
    python random_baseline.py \
        --target-pdb test/5M2J.pdb --target-chain A \
        --epitope-indices 66,67,68,79,80,81,82,83,84,117,118 \
        --framework antitnf --n-random 50 --seed 42 \
        --out /tmp/v9_designs/TNFA_5M2J_random.json
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_target_pdb import setup_target_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"

AA20 = "ARNDCQEGHILKMFPSTWYV"
AA20_NO_CYS = "ARNDQEGHILKMFPSTWYV"  # no Cys in random CDRs (avoids disulfide artifacts)


def load_model():
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(MODEL_PATH)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(MODEL_PATH, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def random_cdr_seq(cdr_length: int, rng: np.random.RandomState) -> str:
    """Generate a random CDR sequence using natural AA frequencies (no Cys)."""
    return "".join(rng.choice(list(AA20_NO_CYS), size=cdr_length))


def inject(cdr_subseq: str, template: str, cdr_indices: list[int]) -> str:
    chars = list(template)
    for i, c in zip(cdr_indices, cdr_subseq):
        chars[i] = c
    return "".join(chars)


def fold_one(model, binder_seq: str, target_seq: str, num_loops=3, num_sampling=14):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=1,
            calculate_confidence=True,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-pdb", required=True)
    p.add_argument("--target-chain", required=True)
    p.add_argument("--epitope-indices", required=True)
    p.add_argument("--framework", default="b5")
    p.add_argument("--n-random", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--fast", action="store_true", help="Use Fast model (quicker)")
    args = p.parse_args()

    epi = [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    rng = np.random.RandomState(args.seed)

    setup = setup_target_design(
        pdb_path=args.target_pdb,
        target_chain=args.target_chain,
        epitope_indices=epi,
        framework=args.framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    cdr_indices = setup["cdr_indices"]
    cdr_len = len(cdr_indices)
    wt_cdr = "".join(binder_wt[i] for i in cdr_indices)

    print(f"=== Random CDR baseline: {cdr_len} CDR positions, "
          f"{args.n_random} random sequences ===")
    print(f"  WT CDR: {wt_cdr}")
    print(f"  Target: {len(target_seq)}aa  Binder: {len(binder_template)}aa")
    print(f"  Epitope: {len(epi)} residues")

    model = load_model()
    results = []

    # Always fold WT first
    print(f"\n  {'Name':>16}  {'pTM':>6}  {'ipTM':>6}  CDR_seq")
    print(f"  {'-'*70}")
    t0 = time.time()
    out = fold_one(model, binder_wt, target_seq)
    wt_ptm = float(out["ptm"][0].item())
    wt_iptm = float(out["iptm"][0].item())
    dt = time.time() - t0
    print(f"  {'WT':>16}  {wt_ptm:>6.3f}  {wt_iptm:>6.3f}  {wt_cdr}  [{dt:.0f}s]")
    results.append({"name": "WT", "iptm": wt_iptm, "ptm": wt_ptm, "cdr_seq": wt_cdr})

    iptm_values = []
    for i in range(args.n_random):
        rcdr = random_cdr_seq(cdr_len, rng)
        full = inject(rcdr, binder_template, cdr_indices)
        try:
            t0 = time.time()
            out = fold_one(model, full, target_seq)
            iptm = float(out["iptm"][0].item())
            ptm = float(out["ptm"][0].item())
            dt = time.time() - t0
            iptm_values.append(iptm)
            results.append({
                "name": f"random_{i+1}",
                "iptm": iptm, "ptm": ptm,
                "cdr_seq": rcdr,
            })
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {'random_' + str(i+1):>16}  {ptm:>6.3f}  {iptm:>6.3f}  "
                      f"{rcdr}  [{dt:.0f}s]  "
                      f"[{i+1}/{args.n_random}, median iptm={np.median(iptm_values):.3f}]",
                      flush=True)
        except Exception as e:
            print(f"  [ERR] random_{i+1}: {e}")
            continue

    iptm_arr = np.array(iptm_values)
    summary = {
        "target_pdb": args.target_pdb,
        "target_chain": args.target_chain,
        "framework": args.framework,
        "cdr_len": cdr_len,
        "n_random": args.n_random,
        "wt_cdr": wt_cdr,
        "wt_iptm": wt_iptm,
        "wt_ptm": wt_ptm,
        "random_iptm_mean": float(iptm_arr.mean()),
        "random_iptm_median": float(np.median(iptm_arr)),
        "random_iptm_std": float(iptm_arr.std()),
        "random_iptm_min": float(iptm_arr.min()),
        "random_iptm_max": float(iptm_arr.max()),
        "random_iptm_p90": float(np.percentile(iptm_arr, 90)),
        "results": results,
    }

    print(f"\n=== Summary ===")
    print(f"  WT             iptm = {wt_iptm:.4f}")
    print(f"  Random (n={len(iptm_values)})  iptm = "
          f"{summary['random_iptm_median']:.4f} ± {summary['random_iptm_std']:.4f}  "
          f"[{summary['random_iptm_min']:.4f}, {summary['random_iptm_max']:.4f}]")
    print(f"  WT z-score = "
          f"{(wt_iptm - summary['random_iptm_mean']) / max(summary['random_iptm_std'], 1e-8):.1f}σ")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
