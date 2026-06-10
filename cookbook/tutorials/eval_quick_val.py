"""Batch Full-ESMFold2 evaluator for the quick-validation panel.

Loads each /tmp/quick_val/runs/<tag>_seed<N>.json, folds every snapshot
with the Full ESMFold2 (1.3G) model, and writes results to
/tmp/quick_val/evals/<tag>_seed<N>_eval.json. Also evaluates the WT
init (--include-wt) for each target as a baseline.

Per snapshot we record pTM, ipTM, CDR→epitope min distance, and all
loss components. Final summary aggregates per-target across seeds.
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


def evaluate_snapshots(
    model,
    snapshots: list[dict],
    target_seq: str,
    binder_template: str,
    binder_wt: str,
    cdr_indices: list[int],
    epitope_indices: list[int],
    prior_bins,
    prior_mask,
    target_len: int,
    binder_len: int,
    include_wt: bool = True,
):
    cdr_to_eval = [(s["cdr_seq"], f"step{s['step']:03d}", s["step"]) for s in snapshots]
    if include_wt:
        wt_cdrs = "".join(binder_wt[i] for i in cdr_indices)
        cdr_to_eval.insert(0, (wt_cdrs, "WT (init)", -1))

    results = []
    print(f"\n  {'Name':>14}  {'pTM':>6}  {'ipTM':>6}  {'CDR→epi':>8}  "
          f"{'inter':>7}  {'intra':>7}  {'epi':>7}  {'prior':>7}  CDR_seq")
    print(f"  {'-'*120}")
    for cdr_sub, name, step in cdr_to_eval:
        if len(cdr_sub) != len(cdr_indices):
            print(f"  [SKIP] {name} cdr length {len(cdr_sub)} != {len(cdr_indices)}")
            continue
        full = inject(cdr_sub, binder_template, cdr_indices)
        if len(full) != binder_len:
            print(f"  [SKIP] {name} full length {len(full)} != {binder_len}")
            continue
        try:
            t0 = time.time()
            out = fold_one(model, full, target_seq, 3, 14)
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
            epitope_token_indices=epitope_indices, cdr_indices=cdr_indices,
            prior_bins=prior_bins[:L_p, :L_p], prior_mask=prior_mask[:L_p, :L_p],
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        midpoints = get_mid_points().to(disto.device)
        probs = torch.softmax(disto, dim=-1)
        e_dist = (probs * midpoints).sum(-1)[0]
        cross = e_dist[target_len:, :target_len]
        cdr_to_e = cross[cdr_indices][:, epitope_indices]
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
    return results


def aggregate_per_target(per_seed_results: list[dict]) -> dict:
    """Compute per-target statistics across seeds.

    For each step (or WT), compute median/mean/std of iptm, pTM, cdr_to_epi.
    """
    by_step: dict[int, list[dict]] = {}
    for r in per_seed_results:
        for x in r["results"]:
            by_step.setdefault(x["step"], []).append(x)

    summary = []
    for step in sorted(by_step.keys()):
        rs = by_step[step]
        iptms = [x["iptm"] for x in rs if x["iptm"] is not None]
        ptms = [x["ptm"] for x in rs if x["ptm"] is not None]
        cdr_min = [x["cdr_to_epi_min"] for x in rs]
        summary.append({
            "step": step,
            "n": len(rs),
            "iptm_median": float(np.median(iptms)) if iptms else None,
            "iptm_mean": float(np.mean(iptms)) if iptms else None,
            "iptm_std": float(np.std(iptms)) if iptms else None,
            "iptm_min": float(np.min(iptms)) if iptms else None,
            "iptm_max": float(np.max(iptms)) if iptms else None,
            "ptm_median": float(np.median(ptms)) if ptms else None,
            "ptm_mean": float(np.mean(ptms)) if ptms else None,
            "cdr_to_epi_median": float(np.median(cdr_min)),
            "cdr_to_epi_mean": float(np.mean(cdr_min)),
        })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/tmp/quick_val/runs")
    p.add_argument("--out-dir", default="/tmp/quick_val/evals")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--require-complete", action="store_true",
                   help="Skip runs whose last snapshot is not at the final step "
                        "(avoids reading partial writes from a running design loop).")
    p.add_argument("--final-step", type=int, default=100,
                   help="Step value considered 'complete' (default 100).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = sorted(glob.glob(f"{args.runs_dir}/*.json"))
    print(f"Found {len(runs)} run files")
    if not runs:
        return

    # Group by target tag (strip _seedN suffix)
    by_tag: dict[str, list[str]] = {}
    skipped_partial = 0
    for r in runs:
        name = Path(r).stem  # e.g. PDL1_5JDS_KN035_seed0
        # tag = everything before _seed
        tag = "_seed".join(name.split("_seed")[:-1])
        if args.require_complete:
            try:
                with open(r) as f:
                    d = json.load(f)
                snaps = d.get("snapshots", [])
                if not snaps or snaps[-1].get("step") != args.final_step:
                    skipped_partial += 1
                    print(f"  [skip-partial] {Path(r).name} last_step="
                          f"{snaps[-1].get('step') if snaps else 'none'}")
                    continue
            except json.JSONDecodeError:
                skipped_partial += 1
                print(f"  [skip-corrupt] {Path(r).name} (read while writing)")
                continue
        by_tag.setdefault(tag, []).append(r)
    if args.require_complete and skipped_partial:
        print(f"Skipped {skipped_partial} partial/corrupt files")
    print(f"Targets: {list(by_tag.keys())}")

    # Load model once
    model = load_model()

    # Target configs (must match run_quick_val.sh)
    target_configs = {
        "PDL1_5JDS_KN035": dict(pdb="test/5JDS.pdb", chain="A",
                                epi=[36,38,43,45,48,50,97,98,99,101,102,103,104,105],
                                fw="kn035"),
        "RBD_6WAQ_VHH72": dict(pdb="test/6WAQ.pdb", chain="B",
                                epi=[35,36,37,38,39,40,41,42,43,44,45,46,49,50],
                                fw="vhh72"),
        "TNFA_5M2J_ANTITNF": dict(pdb="test/5M2J.pdb", chain="A",
                                  epi=[66,67,68,79,80,81,82,83,84,117,118],
                                  fw="antitnf"),
        "RBD_6ZXN_TY1": dict(pdb="/tmp/quick_val/6ZXN_RBD.pdb", chain="A",
                             epi=[95,96,97,100,101,102,194,195,196,197,198,199,200,201,216,218,219,220,228,229,230,231,232,233,234,235,236,237,238,240,241,242,243,244],
                             fw="ty1"),
        "TNFA_5M2M_VHH3": dict(pdb="test/5M2M.pdb", chain="B",
                               epi=[13,14,15,16,17,18,57,58,59,60,61,62,63,64,65,67,94,97,98,100,101,104,105,106,130,131,132,133,134,135,136,137,138,139],
                               fw="vhh3"),
    }

    all_per_target: dict[str, dict] = {}
    for tag, snap_files in by_tag.items():
        if tag not in target_configs:
            print(f"\n[WARN] Unknown tag {tag}, skipping")
            continue
        cfg = target_configs[tag]
        print(f"\n========= {tag} =========")
        setup = setup_target_design(
            pdb_path=cfg["pdb"], target_chain=cfg["chain"],
            epitope_indices=cfg["epi"], framework=cfg["fw"],
        )
        target_seq = setup["target_sequence"]
        binder_template = setup["binder_template"]
        binder_wt = setup["binder_full_sequence"]
        target_len = len(target_seq)
        binder_len = len(binder_template)
        cdr = setup["cdr_indices"]
        epi = setup["epitope_token_indices"]
        prior_bins = setup["prior_bins"]
        prior_mask = setup["prior_mask"]

        per_seed = []
        for sf in snap_files:
            out_path = Path(args.out_dir) / (Path(sf).stem + "_eval.json")
            if args.skip_existing and out_path.exists():
                print(f"  [SKIP] {out_path.name} exists")
                with open(out_path) as f:
                    per_seed.append(json.load(f))
                continue
            print(f"\n  Eval {Path(sf).name}")
            with open(sf) as f:
                d = json.load(f)
            results = evaluate_snapshots(
                model, d["snapshots"], target_seq, binder_template, binder_wt,
                cdr, epi, prior_bins, prior_mask, target_len, binder_len,
                include_wt=True,
            )
            record = {
                "tag": tag, "seed_file": Path(sf).name,
                "target_pdb": cfg["pdb"], "target_chain": cfg["chain"],
                "epitope": epi, "framework": cfg["fw"],
                "binder_len": binder_len, "target_len": target_len,
                "results": results,
            }
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            per_seed.append(record)

        summary = aggregate_per_target(per_seed)
        all_per_target[tag] = {"per_seed": per_seed, "summary": summary}
        print(f"\n=== Per-step aggregate for {tag} ===")
        print(f"  {'step':>6}  {'n':>3}  {'iptm_med':>9}  {'iptm_std':>9}  "
              f"{'ptm_med':>8}  {'cdr2e_med':>10}")
        for s in summary:
            iptm = f"{s['iptm_median']:.3f}" if s['iptm_median'] is not None else "  N/A"
            std = f"{s['iptm_std']:.3f}" if s['iptm_std'] is not None else "  N/A"
            ptm = f"{s['ptm_median']:.3f}" if s['ptm_median'] is not None else "  N/A"
            print(f"  {s['step']:>6}  {s['n']:>3}  {iptm:>9}  {std:>9}  "
                  f"{ptm:>8}  {s['cdr_to_epi_median']:>10.2f}")

    # Write cross-target summary
    out_summary = Path(args.out_dir) / "_all_targets_summary.json"
    with open(out_summary, "w") as f:
        json.dump({tag: {"summary": v["summary"]} for tag, v in all_per_target.items()},
                  f, indent=2)
    print(f"\n\nWrote per-target summary to {out_summary}")


if __name__ == "__main__":
    main()
