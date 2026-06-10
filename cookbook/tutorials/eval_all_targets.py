"""Comprehensive evaluation: Full ESMFold2 on all v9 design snapshots + random baselines.

Reads /tmp/v9_designs/<TAG>_seed<N>_snaps.json, folds top snapshots
with Full ESMFold2, and produces a summary table for the preprint.

Usage:
    python eval_all_targets.py --n-top 5 --out /tmp/v9_designs/SUMMARY.json
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    compute_structure_losses, get_mid_points,
)
from test_target_pdb import setup_target_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"

# Target registry
TARGETS = {
    "RBD_6ZXN_TY1": {
        "pdb": "/tmp/6ZXN_RBD.pdb", "chain": "A",
        "epi": "18,19,22,116,117,118,119,120,122,138,140,152,153,154,155,156,159,160,162,163,164",
        "framework": "ty1", "label": "RBD/Ty1 (6ZXN)", "extra": "",
    },
    "RBD_6WAQ_VHH72": {
        "pdb": "../../test/6WAQ.pdb", "chain": "B",
        "epi": "35,36,37,38,39,40,41,42,43,44,45,46,49,50",
        "framework": "vhh72", "label": "RBD/VHH72 (6WAQ)", "extra": "",
    },
    "PDL1_5JDS": {
        "pdb": "../../test/5JDS.pdb", "chain": "A",
        "epi": "36,38,43,45,48,50,97,98,99,101,102,103,104,105",
        "framework": "kn035", "label": "PD-L1/KN035 (5JDS)", "extra": "--allow-cdr-cys",
    },
    "TNFA_5M2J": {
        "pdb": "../../test/5M2J.pdb", "chain": "A",
        "epi": "66,67,68,79,80,81,82,83,84,117,118",
        "framework": "antitnf", "label": "TNFα/anti-TNF (5M2J)", "extra": "",
    },
    "TNFA_5M2M": {
        "pdb": "../../test/5M2M.pdb", "chain": "A",
        "epi": "15,16,17,18,58,59,60,61,62,63,99,100,130,131,132,133,135",
        "framework": "vhh3", "label": "TNFα/VHH3 (5M2M)", "extra": "",
    },
}


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


def inject(cdr_subseq: str, template: str, cdr_indices: list[int]) -> str:
    chars = list(template)
    for i, c in zip(cdr_indices, cdr_subseq):
        chars[i] = c
    return "".join(chars)


def expected_distance(disto):
    midpoints = get_mid_points().to(disto.device)
    probs = torch.softmax(disto, dim=-1)
    return (probs * midpoints).sum(-1)


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


def eval_one(model, binder_seq, target_seq, binder_len, target_len,
             cdr_indices, epi_indices, prior_bins, prior_mask):
    out = fold_one(model, binder_seq, target_seq)
    disto_bf = out["distogram_logits"].float()
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L),
                      torch.arange(0, binder_len)])
    disto = disto_bf[:, perm, :, :][:, :, perm, :]
    L_p = prior_bins.size(0)
    if disto.size(1) != L_p:
        disto = disto[:, :L_p, :L_p, :]

    e_dist = expected_distance(disto)[0]
    cross = e_dist[target_len:, :target_len]
    cdr_to_e = cross[cdr_indices][:, epi_indices]
    cdr_min = cdr_to_e.min(dim=-1).values.mean().item()

    iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
    return iptm, ptm, cdr_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-top", type=int, default=5, help="Top designs per seed to evaluate")
    p.add_argument("--out", default="/tmp/v9_designs/SUMMARY.json")
    p.add_argument("--targets", nargs="*", default=list(TARGETS),
                   help="Target tags to evaluate (default: all)")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--include-random", action="store_true")
    args = p.parse_args()

    model = load_model()
    all_summaries = {}

    for tag in args.targets:
        if tag not in TARGETS:
            print(f"Unknown target: {tag}")
            continue
        cfg = TARGETS[tag]
        epi = [int(x) for x in cfg["epi"].split(",") if x.strip()]
        print(f"\n{'='*70}")
        print(f"Target: {cfg['label']} ({tag})")
        print(f"{'='*70}")

        setup = setup_target_design(
            pdb_path=cfg["pdb"], target_chain=cfg["chain"],
            epitope_indices=epi, framework=cfg["framework"],
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

        # Evaluate WT
        wt_cdr = "".join(binder_wt[i] for i in cdr_idx)
        print(f"  WT CDR: {wt_cdr} ({len(cdr_idx)}aa)")
        t0 = time.time()
        wt_iptm, wt_ptm, wt_cdr_epi = eval_one(
            model, binder_wt, target_seq, binder_len, target_len,
            cdr_idx, epi_idx, prior_bins, prior_mask)
        print(f"  WT: iptm={wt_iptm:.3f}  ptm={wt_ptm:.3f}  "
              f"cdr→epi={wt_cdr_epi:.1f}Å  [{time.time()-t0:.0f}s]")

        per_seed_results = []
        for seed in args.seeds:
            snap_path = f"/tmp/v9_designs/{tag}_seed{seed}_snaps.json"
            if not os.path.exists(snap_path):
                print(f"  [SKIP] seed {seed}: no snapshot file")
                continue

            with open(snap_path) as f:
                snap_data = json.load(f)
            snaps = snap_data.get("snapshots", [])
            if not snaps:
                print(f"  [SKIP] seed {seed}: empty snapshots")
                continue

            print(f"  Seed {seed}: {len(snaps)} snapshots, "
                  f"init CDR: {snap_data.get('init_cdr', '?')}")

            # Find best snapshots by inter loss (design-time metric)
            ranked = sorted(snaps, key=lambda s: s.get("inter", float("inf")))
            top_n = ranked[:args.n_top]

            seed_evals = []
            for snap in top_n:
                cdr_sub = snap["cdr_seq"]
                if len(cdr_sub) != len(cdr_idx):
                    print(f"    [SKIP] step {snap['step']}: CDR len mismatch")
                    continue
                full = inject(cdr_sub, binder_template, cdr_idx)
                try:
                    t0 = time.time()
                    iptm, ptm, cdr_epi = eval_one(
                        model, full, target_seq, binder_len, target_len,
                        cdr_idx, epi_idx, prior_bins, prior_mask)
                    dt = time.time() - t0
                    ev = {
                        "step": snap["step"],
                        "cdr_seq": cdr_sub,
                        "iptm": iptm, "ptm": ptm,
                        "cdr_to_epi": cdr_epi,
                        "design_inter": snap.get("inter"),
                        "time_s": dt,
                    }
                    seed_evals.append(ev)
                    print(f"    step {snap['step']:3d}: iptm={iptm:.3f}  "
                          f"ptm={ptm:.3f}  cdr→epi={cdr_epi:.1f}Å  "
                          f"{cdr_sub}  [{dt:.0f}s]")
                except Exception as e:
                    print(f"    [ERR] step {snap['step']}: {e}")

            per_seed_results.append({"seed": seed, "evals": seed_evals})

            # Find best iptm for this seed
            if seed_evals:
                best = max(seed_evals, key=lambda e: e["iptm"] or -1)
                print(f"  Seed {seed} best: iptm={best['iptm']:.3f}  "
                      f"Δiptm={best['iptm'] - wt_iptm:+.3f}  "
                      f"step={best['step']}")

        # Summary across seeds
        all_iptm = []
        for sr in per_seed_results:
            for ev in sr["evals"]:
                if ev["iptm"] is not None:
                    all_iptm.append(ev["iptm"])

        best_iptm = max(all_iptm) if all_iptm else None
        median_iptm = float(np.median(all_iptm)) if all_iptm else None
        summary = {
            "tag": tag, "label": cfg["label"],
            "wt_iptm": wt_iptm, "wt_ptm": wt_ptm,
            "wt_cdr_to_epi": wt_cdr_epi,
            "best_design_iptm": best_iptm,
            "median_design_iptm": median_iptm,
            "delta_iptm_best": (best_iptm - wt_iptm) if best_iptm is not None else None,
            "delta_iptm_median": (median_iptm - wt_iptm) if median_iptm is not None else None,
            "n_evals": len(all_iptm),
            "per_seed": per_seed_results,
        }
        all_summaries[tag] = summary

        print(f"\n  {'─'*60}")
        print(f"  WT:     iptm={wt_iptm:.3f}  cdr→epi={wt_cdr_epi:.1f}Å")
        print(f"  Best:   iptm={best_iptm:.3f}  (Δ={summary['delta_iptm_best']:+.3f})")
        print(f"  Median: iptm={median_iptm:.3f}  (Δ={summary['delta_iptm_median']:+.3f})")

    # ---- Final summary table ----
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    header = (f"  {'Target':<24} {'WT ipTM':>8} {'Best ipTM':>10} "
              f"{'Δ (best)':>9} {'Δ (med)':>9} {'N':>5}")
    print(header)
    print(f"  {'-'*70}")
    for tag in args.targets:
        s = all_summaries.get(tag)
        if s is None:
            continue
        print(f"  {s['label']:<24} {s['wt_iptm']:>8.3f} "
              f"{s['best_design_iptm']:>10.3f} "
              f"{s['delta_iptm_best']:>+9.3f} "
              f"{s['delta_iptm_median']:>+9.3f} "
              f"{s['n_evals']:>5}")

    with open(args.out, "w") as f:
        # Convert numpy values
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            return obj
        json.dump(convert(all_summaries), f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
