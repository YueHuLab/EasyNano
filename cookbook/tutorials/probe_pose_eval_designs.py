"""Step 4 eval: After v9 iterative design, evaluate best per-seed CDRs with
multi-sample averaging and compare to v9_best_15seed_p116Y baseline.

Reads:
  /tmp/b5_iter_p116Y/snaps_seed*.json   (per-seed design snapshots)

For each seed, find the best iptm step's CDR. Then for each top CDR:
  - Run num_diffusion_samples=3 × 3 seeds (= 9 effective samples)
  - Compute median, mean, std
  - Compare to v9_best_15seed_p116Y baseline (median 0.692, std 0.020)

Outputs a sorted ranking and saves to /tmp/b5_iter_p116Y/eval.json
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_b5_pdb import setup_design  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127
N_EVAL_SEEDS = 3
N_DIFF_SAMPLES = 3

# Framework pieces
PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"  # 25 chars
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"  # 19 chars
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"  # 41 chars
POST_H3 = "WGQGTLVTVS"  # 10 chars

# CDR positions (0-indexed): 25-34, 54-59, 101-116
CDR_H1 = list(range(25, 35))  # 10
CDR_H2 = list(range(54, 60))  # 6
CDR_H3 = list(range(101, 117))  # 16
CDR_INDICES = CDR_H1 + CDR_H2 + CDR_H3


def extract_cdr_from_seq(full_seq: str) -> str:
    return "".join(full_seq[i] for i in CDR_INDICES)


def make_full_seq(cdr32: str) -> str:
    return (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
            + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)


def load_model():
    print("\nLoading FULL ESMFold2 (1.3G) ...", flush=True)
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


def fold_one(model, binder_seq, target_seq, seed, n_diff_samples):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=3, num_sampling_steps=14,
            num_diffusion_samples=n_diff_samples, calculate_confidence=True,
        )
    dt = time.time() - t0
    iptm = out["iptm"].float().cpu().numpy()
    return {"iptm_per_sample": iptm.tolist(), "time_s": dt}


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}", flush=True)

    out_dir = Path("/tmp/b5_iter_p116Y")
    snap_files = sorted(out_dir.glob("snaps_seed*.json"))
    print(f"Found {len(snap_files)} snapshot files", flush=True)

    # Find best CDR per seed (highest iptm at any step)
    best_per_seed = []
    for sf in snap_files:
        seed = int(sf.stem.split("seed")[-1])
        with open(sf) as f:
            snaps = json.load(f)
        if not snaps:
            print(f"  seed {seed}: empty snaps, skipping", flush=True)
            continue
        # Find step with max iptm
        best_step = max(snaps.keys(),
                        key=lambda k: snaps[k].get("iptm", 0) or 0)
        best_iptm = snaps[best_step].get("iptm")
        best_seq = snaps[best_step].get("seq", "")
        best_cdr = extract_cdr_from_seq(best_seq) if best_seq else ""
        best_per_seed.append({
            "seed": seed, "best_step": best_step, "best_iptm": best_iptm,
            "best_cdr": best_cdr, "best_full_seq": best_seq,
        })
        print(f"  seed {seed}: best_iptm={best_iptm:.3f}@{best_step}  "
              f"CDR={best_cdr}", flush=True)

    # Also include the init (v9_best_15seed_p116Y) as a baseline reference
    BASELINE_CDR = "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY"

    # Build candidate list
    candidates = [("baseline_p116Y", BASELINE_CDR, "v9_best_15seed_p116Y init")]
    for bp in best_per_seed:
        name = f"seed{bp['seed']}_step{bp['best_step']}"
        candidates.append((name, bp["best_cdr"],
                          f"from iter design (iptm {bp['best_iptm']:.3f})"))

    print(f"\n{'='*70}", flush=True)
    print(f"=== Step 4 EVAL: {len(candidates)} candidates × "
          f"{N_EVAL_SEEDS * N_DIFF_SAMPLES} samples each ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    model = load_model()
    seeds = list(range(N_EVAL_SEEDS))

    eval_results = {}
    for name, cdr, desc in candidates:
        full_seq = make_full_seq(cdr)
        eval_results[name] = {"cdr": cdr, "desc": desc,
                              "iptm_all_samples": []}
        print(f"--- {name} ---", flush=True)
        print(f"  CDR: {cdr}  ({desc})", flush=True)
        for s in seeds:
            r = fold_one(model, full_seq, target_seq, seed=s,
                         n_diff_samples=N_DIFF_SAMPLES)
            eval_results[name]["iptm_all_samples"].extend(r["iptm_per_sample"])
            print(f"  seed={s}  iptm={[f'{x:.3f}' for x in r['iptm_per_sample']]}  "
                  f"({r['time_s']:.1f}s)", flush=True)
        all_samples = eval_results[name]["iptm_all_samples"]
        print(f"  → all samples (n={len(all_samples)}):  "
              f"med={np.median(all_samples):.3f}  "
              f"mean={np.mean(all_samples):.3f}  "
              f"std={np.std(all_samples):.3f}  "
              f"min={min(all_samples):.3f}  max={max(all_samples):.3f}",
              flush=True)
        print(flush=True)

    # === Final summary ===
    print(f"\n{'='*70}", flush=True)
    print("=== STEP 4 FINAL RANKING ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    summary = []
    for name, _, _ in candidates:
        all_samples = eval_results[name]["iptm_all_samples"]
        summary.append({
            "name": name,
            "cdr": eval_results[name]["cdr"],
            "desc": eval_results[name]["desc"],
            "n": len(all_samples),
            "median": float(np.median(all_samples)),
            "mean": float(np.mean(all_samples)),
            "std": float(np.std(all_samples)),
            "min": float(min(all_samples)),
            "max": float(max(all_samples)),
        })

    summary.sort(key=lambda r: r["median"], reverse=True)
    print(f"{'name':28s}  {'n':>3s}  {'med':>5s}  {'mean':>5s}  "
          f"{'std':>5s}  {'min':>5s}  {'max':>5s}  CDR", flush=True)
    print("-" * 130, flush=True)
    for s in summary:
        print(f"{s['name']:28s}  {s['n']:3d}  {s['median']:5.3f}  "
              f"{s['mean']:5.3f}  {s['std']:5.3f}  "
              f"{s['min']:5.3f}  {s['max']:5.3f}  {s['cdr']}", flush=True)

    # Compare to baseline
    base = next((s for s in summary if s["name"] == "baseline_p116Y"), None)
    if base:
        print(f"\n  baseline_p116Y: med={base['median']:.3f} std={base['std']:.3f}")
        for s in summary:
            if s["name"] == "baseline_p116Y":
                continue
            delta = s["median"] - base["median"]
            std_ratio = s["std"] / base["std"] if base["std"] > 0 else float("inf")
            marker = ""
            if delta > 0.02 and s["std"] < 2 * base["std"]:
                marker = " ← IMPROVED"
            elif delta < -0.02:
                marker = " ← REGRESSED"
            print(f"  {s['name']:28s} Δ_med={delta:+.3f}  "
                  f"std_ratio={std_ratio:.2f}x{marker}", flush=True)

    out = {
        "init_seq": make_full_seq(BASELINE_CDR),
        "n_eval_seeds": N_EVAL_SEEDS,
        "n_diff_samples": N_DIFF_SAMPLES,
        "best_per_seed": best_per_seed,
        "candidates": [
            {"name": n, "cdr": c, "desc": d} for n, c, d in candidates
        ],
        "eval_results": eval_results,
        "summary_sorted_by_median": summary,
    }
    with open(out_dir / "eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/eval.json", flush=True)


if __name__ == "__main__":
    main()
