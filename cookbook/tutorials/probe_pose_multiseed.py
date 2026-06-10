"""Step 1 of plan: Multi-seed iptm evaluation on top v9/v16 candidates.

Question: How reliable is the iptm signal? Is the leaderboard stable under seed variation?

For each candidate sequence we run N_SEEDS folds, compute iptm median/mean/std,
ipSAE descriptors, and rank stability. Then we can tell whether a candidate is
"real" (robustly high) or "lucky" (one good seed out of five).
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
from design_b5_mps_v9_cacoord import extract_ca_per_token  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127
N_SEEDS = 5

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"
POST_H3 = "WGQGTLVTVS"


def make_full_seq(cdr32: str) -> str:
    assert len(cdr32) == 32
    return (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
            + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)


# Top candidates — order matches the v9/v16 leaderboards
CANDIDATES = [
    ("v9_best_15seed",  "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR", 0.717),
    ("v9_step48",       "GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR", 0.619),
    ("v16_s5_s44",      "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR", 0.572),
    ("v16_s5_s56",      "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR", 0.539),
    ("v16_s2_s44",      "GLQIGYGWYMSYSGQSRVVTDSSTPIYKAPIR", 0.527),
    ("v16_init_v2s050", "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR", 0.471),
]


def load_model_full():
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


def fold_one(model, binder_seq, target_seq, seed):
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
            num_diffusion_samples=1, calculate_confidence=True,
        )
    dt = time.time() - t0
    iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
    pae = out["pae"][0].float().cpu().numpy()
    return {"iptm": iptm, "ptm": ptm, "pae": pae, "time_s": dt}


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    epi_in_pae = [e + BINDER_LEN for e in epi]
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}", flush=True)
    print(f"epitope (21): {epi}", flush=True)
    print(f"CDR (32): {cdr[:10]} | {cdr[10:16]} | {cdr[16:]}", flush=True)

    out_dir = Path("/tmp/b5_multiseed")
    out_dir.mkdir(exist_ok=True)

    model = load_model_full()

    # Per-seed: 0..N-1
    seeds = list(range(N_SEEDS))

    # Store all iptm per (candidate, seed)
    results = {}  # name -> {seed: iptm}
    ptdm_results = {}
    ipsae_p10 = {}

    print(f"\n{'='*70}", flush=True)
    print(f"=== Step 1: Multi-seed iptm evaluation ({N_SEEDS} seeds × "
          f"{len(CANDIDATES)} candidates) ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    for name, cdr_seq, known_iptm in CANDIDATES:
        full_seq = make_full_seq(cdr_seq)
        results[name] = {}
        ptdm_results[name] = {}
        ipsae_p10[name] = {}
        print(f"--- {name} ---", flush=True)
        print(f"  CDR : {cdr_seq} (known iptm={known_iptm:.3f})", flush=True)
        for s in seeds:
            t0 = time.time()
            r = fold_one(model, full_seq, target_seq, seed=s)
            pae_cdr_epi = r["pae"][np.ix_(cdr, epi_in_pae)]
            ipsae_min = float(pae_cdr_epi.min())
            ipsae_p10v = float(np.percentile(pae_cdr_epi, 10))
            results[name][s] = r["iptm"]
            ptdm_results[name][s] = r["ptm"]
            ipsae_p10[name][s] = ipsae_p10v
            print(f"  seed={s}  iptm={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
                  f"ipSAE_p10={ipsae_p10v:.2f}  ({r['time_s']:.1f}s)",
                  flush=True)
        iptms = list(results[name].values())
        ptms = list(ptdm_results[name].values())
        ips = list(ipsae_p10[name].values())
        print(f"  → iptm: median={np.median(iptms):.3f}  "
              f"mean={np.mean(iptms):.3f}  std={np.std(iptms):.3f}  "
              f"min={min(iptms):.3f}  max={max(iptms):.3f}", flush=True)
        print(f"  → pTM:  median={np.median(ptms):.3f}  "
              f"mean={np.mean(ptms):.3f}", flush=True)
        print(f"  → ipSAE_p10: median={np.median(ips):.2f}  "
              f"min={min(ips):.2f}", flush=True)
        print(flush=True)

    # === Summary ranking ===
    print(f"\n{'='*70}", flush=True)
    print("=== SUMMARY (sorted by median iptm) ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    summary = []
    for name, _, known in CANDIDATES:
        iptms = list(results[name].values())
        ips = list(ipsae_p10[name].values())
        ptms = list(ptdm_results[name].values())
        summary.append({
            "name": name,
            "known_single_seed_iptm": known,
            "iptm_median": float(np.median(iptms)),
            "iptm_mean": float(np.mean(iptms)),
            "iptm_std": float(np.std(iptms)),
            "iptm_min": float(min(iptms)),
            "iptm_max": float(max(iptms)),
            "ptm_median": float(np.median(ptms)),
            "ipsae_p10_median": float(np.median(ips)),
            "ipsae_p10_min": float(min(ips)),
        })

    summary.sort(key=lambda r: r["iptm_median"], reverse=True)
    print(f"{'name':22s}  {'med':>5s}  {'mean':>5s}  {'std':>5s}  "
          f"{'min':>5s}  {'max':>5s}  {'known':>5s}  {'ipSAE_p10_med':>13s}",
          flush=True)
    print("-" * 80, flush=True)
    for s in summary:
        print(f"{s['name']:22s}  {s['iptm_median']:5.3f}  "
              f"{s['iptm_mean']:5.3f}  {s['iptm_std']:5.3f}  "
              f"{s['iptm_min']:5.3f}  {s['iptm_max']:5.3f}  "
              f"{s['known_single_seed_iptm']:5.3f}  "
              f"{s['ipsae_p10_median']:13.2f}", flush=True)

    # Spearman rank stability: known_iptm order vs median order
    from scipy.stats import spearmanr
    known_ordered = sorted(CANDIDATES, key=lambda c: c[2], reverse=True)
    known_ranks = {c[0]: i for i, c in enumerate(known_ordered)}
    median_ranks = {s["name"]: i for i, s in enumerate(summary)}
    ks = [known_ranks[c[0]] for c in CANDIDATES]
    ms = [median_ranks[c[0]] for c in CANDIDATES]
    rho, p = spearmanr(ks, ms)
    print(f"\n  Spearman rank corr (known-seed vs median): ρ={rho:.3f}  "
          f"p={p:.3f}", flush=True)
    if rho > 0.7:
        print("  → Rankings are STABLE: known good sequences stay on top "
              "with median", flush=True)
    elif rho > 0.4:
        print("  → Rankings MODERATELY STABLE: noise reshuffles the middle, "
              "top is OK", flush=True)
    else:
        print("  → Rankings UNSTABLE: seed noise dominates; need more seeds "
              "or rank by median", flush=True)

    # Save
    out = {
        "n_seeds": N_SEEDS,
        "candidates": [
            {"name": n, "cdr": c, "known_iptm": k,
             "iptm_per_seed": results[n],
             "ptm_per_seed": ptdm_results[n],
             "ipsae_p10_per_seed": ipsae_p10[n]}
            for n, c, k in CANDIDATES
        ],
        "summary_sorted_by_median": summary,
        "spearman_known_vs_median": float(rho),
        "spearman_p": float(p),
    }
    with open(out_dir / "multiseed.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/multiseed.json", flush=True)


if __name__ == "__main__":
    main()
