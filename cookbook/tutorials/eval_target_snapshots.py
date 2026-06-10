"""Re-evaluate design snapshots with Full ESMFold2 (1.3G).

Reads a snapshot JSON (from design_target.py) and folds each candidate
binder+target complex with the full-quality model, reporting pTM, ipTM,
CDR-to-epitope distance, and all loss components.

This is the "ranking" step: design used Fast ESMFold2 for cheap
gradients, but the final picks are made with the Full model where
iptm is calibrated.

Usage:
    python eval_target_snapshots.py \\
        --target-pdb test/5JDS.pdb --target-chain A \\
        --epitope-indices 36,38,43,45,48,50,97,98,99,101,102,103,104,105 \\
        --framework b5 \\
        --snapshots /tmp/pdl1_seed0_snaps.json \\
        --out /tmp/pdl1_seed0_eval.json
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

from binder_design_hy_losses import (  # noqa: E402
    compute_structure_losses, get_mid_points,
)
from test_target_pdb import setup_target_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"


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
    print(f"  loaded in {time.time() - t0:.1f}s")
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


def fold_one(model, binder_seq: str, target_seq: str, num_loops: int, num_sampling: int):
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
    p.add_argument("--snapshots", required=True, help="Path to design_target JSON output")
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--num-sampling", type=int, default=14)
    p.add_argument("--out", required=True)
    p.add_argument("--include-wt", action="store_true")
    p.add_argument("--top-k", type=int, default=0,
                   help="Only evaluate top-K snapshots by design-time inter loss "
                        "(0 = all)")
    args = p.parse_args()

    epi = [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    print(f"=== Re-evaluating {args.snapshots} ===\n")
    with open(args.snapshots) as f:
        snap_data = json.load(f)
    snapshots = snap_data["snapshots"]
    if args.top_k > 0:
        snapshots = sorted(snapshots, key=lambda s: s["inter"])[:args.top_k]
    print(f"  {len(snapshots)} snapshots to evaluate (top-k={args.top_k})")
    print(f"  eval config: {args.num_loops}-loops, {args.num_sampling}-sample, "
          f"confidence-on\n")

    setup = setup_target_design(
        pdb_path=args.target_pdb,
        target_chain=args.target_chain,
        epitope_indices=epi,
        framework=args.framework,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    model = load_model()
    results = []
    cdr_to_eval = [(s["cdr_seq"], f"step{s['step']:03d}", s["step"]) for s in snapshots]
    if args.include_wt:
        wt_cdrs = "".join(binder_wt[i] for i in cdr)
        cdr_to_eval.insert(0, (wt_cdrs, "WT (initial)", -1))

    print(f"\n  {'Name':>14}  {'pTM':>6}  {'ipTM':>6}  {'CDR→epi':>8}  "
          f"{'inter':>7}  {'intra':>7}  {'epi':>7}  {'prior':>7}  CDR_seq")
    print(f"  {'-'*120}")
    for cdr_sub, name, step in cdr_to_eval:
        if len(cdr_sub) != len(cdr):
            print(f"  [SKIP] {name} cdr length {len(cdr_sub)} != {len(cdr)}")
            continue
        full = inject(cdr_sub, binder_template, cdr)
        if len(full) != binder_len:
            print(f"  [SKIP] {name} full length {len(full)} != {binder_len}")
            continue
        try:
            t0 = time.time()
            out = fold_one(model, full, target_seq, args.num_loops, args.num_sampling)
            dt = time.time() - t0
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
            continue

        disto_bf = out["distogram_logits"].float()
        L = disto_bf.size(1)
        perm = torch.cat([torch.arange(binder_len, L),
                          torch.arange(0, binder_len)])
        disto = disto_bf[:, perm, :, :][:, :, perm, :]
        L_p = prior_bins.size(0)
        if disto.size(1) != L_p:
            disto = disto[:, :L_p, :L_p, :]
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=prior_bins[:L_p, :L_p], prior_mask=prior_mask[:L_p, :L_p],
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        e_dist = expected_distance(disto)[0]
        cross = e_dist[target_len:, :target_len]
        cdr_to_e = cross[cdr][:, epi]
        cdr_min = cdr_to_e.min(dim=-1).values.mean().item()
        ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
        record = {
            "name": name, "step": step, "cdr_seq": cdr_sub, "full_seq": full,
            "ptm": ptm, "iptm": iptm,
            "cdr_to_epi_min": cdr_min,
            "inter": float(losses["inter_contact_loss"].item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "epi": float(losses["epitope_loss"].item()),
            "prior": float(losses["structure_prior_loss"].item()),
            "total": float(losses["total_loss"].item()),
            "fold_time_s": dt,
        }
        results.append(record)
        ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
        iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
        print(f"  {name:>14}  {ptm_s:>6}  {iptm_s:>6}  {cdr_min:>8.2f}  "
              f"{record['inter']:>7.3f}  {record['intra']:>7.3f}  "
              f"{record['epi']:>7.3f}  {record['prior']:>7.3f}  {cdr_sub}")

    with open(args.out, "w") as f:
        json.dump({
            "target_pdb": args.target_pdb,
            "target_chain": args.target_chain,
            "epitope": epi,
            "framework": args.framework,
            "binder_len": binder_len,
            "target_len": target_len,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved {len(results)} results to {args.out}")

    if results:
        ranked_iptm = sorted(results, key=lambda r: -(r["iptm"] or 0))
        ranked_cdr = sorted(results, key=lambda r: r["cdr_to_epi_min"])
        print(f"\n=== Top 5 by ipTM ===")
        for r in ranked_iptm[:5]:
            print(f"  {r['name']}: ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
                  f"CDR→epi={r['cdr_to_epi_min']:.2f}Å  {r['cdr_seq']}")
        print(f"\n=== Top 5 by CDR→epi min-distance ===")
        for r in ranked_cdr[:5]:
            print(f"  {r['name']}: CDR→epi={r['cdr_to_epi_min']:.2f}Å  "
                  f"ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  {r['cdr_seq']}")


if __name__ == "__main__":
    main()
