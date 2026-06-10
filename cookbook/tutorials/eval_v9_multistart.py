"""Aggregate eval: Full ESMFold2 evaluate all multi-start v9 seed candidates.

Loads each seed's snapshot file, evaluates with Full model, then prints a
unified top-10 by iptm and a top-10 by CDR→epi across all seeds.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    compute_structure_losses, get_mid_points,
)
from test_b5_pdb import setup_design  # noqa: E402

DEVICE = "mps"


def load_model_full():
    print("Loading FULL ESMFold2 (1.3G) for evaluation ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained("/Users/huyue/esm-c-fold2/ESMFold2")
    config.esmc_id = "/Users/huyue/esm-c-fold2/ESMC-6B"
    model = ESMFold2Model.from_pretrained(
        "/Users/huyue/esm-c-fold2/ESMFold2", config=config
    ).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def fold_one(model, binder_seq, target_seq, num_loops=3, num_sampling=14):
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


def dedupe_by_seq(records):
    """Collapse records with identical binder sequences, keeping the first."""
    seen = set()
    out = []
    for r in records:
        if r["full_seq"] in seen:
            continue
        seen.add(r["full_seq"])
        out.append(r)
    return out


def evaluate_multi(snap_paths, out_path, include_v9_step48=True):
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_len = 127
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    target_len = len(target_seq)

    model = load_model_full()

    # Build eval list: v9 step 48 baseline + per-seed snapshots
    eval_list = []
    if include_v9_step48:
        v9_step48 = (
            "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGWYMSLGWFRQAPGQGLEAVAAI"
            "SYSGQRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVADSPQR"
            "IYKAPIRWGQGTLVTVS"
        )
        eval_list.append(("v9 step 48 (best so far)", v9_step48))

    for path in snap_paths:
        with open(path) as f:
            data = json.load(f)
        seed = data.get("config", {}).get("seed", "?")
        init_seq = data["init_full_seq"]
        eval_list.append((f"{Path(path).stem} init (seed={seed})", init_seq))
        for snap in data["snapshots"]:
            name = f"{Path(path).stem} step {snap['step']}"
            eval_list.append((name, snap["full_seq"]))

    # Dedupe by sequence to save fold time
    by_seq = {}
    for name, full in eval_list:
        by_seq.setdefault(full, []).append(name)
    deduped = [(names[0], full) for full, names in by_seq.items()]
    print(f"  Total: {len(eval_list)} candidates → {len(deduped)} unique sequences",
          flush=True)

    print(f"\n  {'Name':>40}  {'pTM':>6}  {'ipTM':>6}  {'CDR→epi':>8}", flush=True)
    print(f"  {'-'*80}", flush=True)
    results = []
    for name, full in deduped:
        cdr_seq = "".join(full[i] for i in cdr)
        try:
            t0 = time.time()
            out = fold_one(model, full, target_seq, num_loops=3, num_sampling=14)
            dt = time.time() - t0
        except Exception as e:
            print(f"  [ERR] {name}: {e}", flush=True)
            continue
        disto_bf = out["distogram_logits"].float()
        L = disto_bf.size(1)
        perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
        disto = disto_bf[:, perm, :, :][:, :, perm, :]
        L_p = prior_bins.size(0)
        if disto.size(1) != L_p:
            disto = disto[:, :L_p, :L_p, :]
        pb, pm = prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb, prior_mask=pm,
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        midpoints = get_mid_points().to(disto.device)
        probs = torch.softmax(disto, dim=-1)
        e_dist = (probs * midpoints).sum(-1)[0]
        cross = e_dist[target_len:, :target_len]
        cdr_to_e = cross[cdr][:, epi]
        cdr_min = cdr_to_e.min(dim=-1).values.mean().item()
        ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
        record = {
            "name": name, "cdr_seq": cdr_seq, "full_seq": full,
            "ptm": ptm, "iptm": iptm,
            "cdr_to_epi_min": cdr_min,
            "inter": float(losses["inter_contact_loss"].item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "epi": float(losses["epitope_loss"].item()),
            "fold_time_s": dt,
        }
        results.append(record)
        ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
        iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
        print(f"  {name:>40}  {ptm_s:>6}  {iptm_s:>6}  {cdr_min:>8.2f}",
              flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {len(results)} unique-sequence results to {out_path}", flush=True)

    print(f"\n=== Top 10 by ipTM (across all seeds) ===", flush=True)
    for r in sorted(results, key=lambda r: -(r["iptm"] or 0))[:10]:
        print(f"  {r['name']:>40}: ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
              f"CDR→epi={r['cdr_to_epi_min']:.2f}Å  "
              f"CDR={r['cdr_seq']}", flush=True)
    print(f"\n=== Top 10 by CDR→epi ===", flush=True)
    for r in sorted(results, key=lambda r: r["cdr_to_epi_min"])[:10]:
        print(f"  {r['name']:>40}: CDR→epi={r['cdr_to_epi_min']:.2f}Å  "
              f"ipTM={r['iptm']:.3f}  CDR={r['cdr_seq']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snaps", nargs="+", required=True,
                   help="snapshot JSON files (one per seed)")
    p.add_argument("--out", default="/tmp/b5_v9_multistart_eval.json")
    args = p.parse_args()
    evaluate_multi(args.snaps, args.out)
