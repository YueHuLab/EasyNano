"""Step 3 of plan: Cross-architecture validation with ESMFold2-Fast.

Step 1 was on FULL ESMFold2 (48 trunk layers, ~1.3G params).
Step 3 repeats the eval on the FAST model (24 trunk layers, ~721M params).

Same family, different config:
  - Both have d_single=384, d_pair=256
  - Fast is half the depth of Full
  - Both share the same ESMC trunk and structure head

If a sequence's iptm ranking is preserved across the two architectures,
the binding signal is more likely to be real, not an artifact of the
specific model instance / config.

Hypothesis: rankings should agree on the top 3-5 candidates (they share
the same conditioning trunk). The FAST model will be lower-quality
(lower iptm values), but the relative ranking should hold.
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
N_SEEDS = 3

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"
POST_H3 = "WGQGTLVTVS"


def make_full_seq(cdr32: str, fw_muts=None) -> str:
    assert len(cdr32) == 32
    seq = (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
           + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)
    if fw_muts:
        s = list(seq)
        for pos, new_aa in fw_muts:
            s[pos] = new_aa
        seq = "".join(s)
    return seq


# Top candidates — combine Step 1 winners with Step 2 improvements
TOP_CANDIDATES = [
    ("v9_best_15seed",       "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR",  []),
    ("v9_best_15seed_p116Y", "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR",  [(116, "Y")]),  # Step 2 winner
    ("v16_s5_s56",           "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR",  []),
    ("v16_s5_s56_p71V",      "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR",  [(71, "V")]),   # Step 2 winner
    ("v16_s5_s44",           "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR",  []),
    ("v16_init_v2s050",      "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR",  []),
]


def load_model(path: str, name: str):
    print(f"\nLoading {name} from {path} ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(path, local_files_only=True)
    config.esmc_id = "/Users/huyue/esm-c-fold2/ESMC-6B"
    model = ESMFold2Model.from_pretrained(
        path, config=config, local_files_only=True
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
    epi_in_pae = np.array([e + BINDER_LEN for e in epi])
    cdr_arr = np.array(cdr)
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}", flush=True)

    out_dir = Path("/tmp/b5_crossval")
    out_dir.mkdir(exist_ok=True)

    # Read Step 1 results to use as reference
    step1_path = Path("/tmp/b5_multiseed/multiseed.json")
    step1_summary = {}
    if step1_path.exists():
        with open(step1_path) as f:
            step1 = json.load(f)
        for c in step1["candidates"]:
            step1_summary[c["name"]] = c
        print("\nStep 1 (Full ESMFold2) reference loaded.", flush=True)

    seeds = list(range(N_SEEDS))
    fast_results = {}

    print(f"\n{'='*70}", flush=True)
    print(f"=== Step 3: Cross-architecture validation ===", flush=True)
    print(f"  Model: ESMFold2-Fast (24 trunk layers, ~721M params)", flush=True)
    print(f"  Candidates: {len(TOP_CANDIDATES)} top from Step 1", flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Total folds: {len(TOP_CANDIDATES) * N_SEEDS}", flush=True)
    print(f"{'='*70}\n", flush=True)

    model = load_model("/Users/huyue/esm-c-fold2/ESMFold2-Fast", "ESMFold2-Fast")

    for name, cdr_seq, fw_muts in TOP_CANDIDATES:
        full_seq = make_full_seq(cdr_seq, fw_muts=fw_muts)
        fast_results[name] = {"fw_muts": fw_muts, "iptm": {}, "ptm": {},
                              "ipsae_p10": {}, "iptm_arr": []}
        print(f"--- {name} ---", flush=True)
        print(f"  CDR: {cdr_seq}", flush=True)
        if fw_muts:
            for p, a in fw_muts:
                print(f"  Framework mutation: pos{p} → {a}", flush=True)
        if name in step1_summary:
            s1 = step1_summary[name]
            print(f"  Step 1 (Full):  median={s1.get('iptm_median', '?')}  "
                  f"std={s1.get('iptm_std', '?')}", flush=True)
        for s in seeds:
            r = fold_one(model, full_seq, target_seq, seed=s)
            pae_cdr_epi = r["pae"][np.ix_(cdr_arr, epi_in_pae)]
            ipsae_p10v = float(np.percentile(pae_cdr_epi, 10))
            fast_results[name]["iptm"][s] = r["iptm"]
            fast_results[name]["ptm"][s] = r["ptm"]
            fast_results[name]["ipsae_p10"][s] = ipsae_p10v
            fast_results[name]["iptm_arr"].append(r["iptm"])
            print(f"  seed={s}  iptm={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
                  f"ipSAE_p10={ipsae_p10v:.2f}  ({r['time_s']:.1f}s)",
                  flush=True)
        iptms = fast_results[name]["iptm_arr"]
        ips = list(fast_results[name]["ipsae_p10"].values())
        print(f"  → iptm: median={np.median(iptms):.3f}  "
              f"mean={np.mean(iptms):.3f}  std={np.std(iptms):.3f}", flush=True)
        print(flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("=== CROSS-VALIDATION SUMMARY (ESMFold2-Fast vs Full) ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    summary = []
    for name, _, _ in TOP_CANDIDATES:
        iptms = fast_results[name]["iptm_arr"]
        summary.append({
            "name": name,
            "fast_iptm_median": float(np.median(iptms)),
            "fast_iptm_mean": float(np.mean(iptms)),
            "fast_iptm_std": float(np.std(iptms)),
            "fast_iptm_min": float(min(iptms)),
            "fast_iptm_max": float(max(iptms)),
        })
        # Match with Step 1 (if applicable)
        if name in step1_summary:
            s1 = step1_summary[name]
            iptms_s1 = list(s1["iptm_per_seed"].values())
            summary[-1]["full_iptm_median"] = float(np.median(iptms_s1))
            summary[-1]["full_iptm_mean"] = float(np.mean(iptms_s1))
            summary[-1]["full_iptm_std"] = float(np.std(iptms_s1))

    summary.sort(key=lambda r: r.get("full_iptm_median", 0), reverse=True)
    print(f"{'name':22s}  {'fast_med':>8s}  {'fast_mean':>9s}  "
          f"{'fast_std':>8s}  {'full_med':>8s}  {'full_std':>8s}  "
          f"{'Δ':>6s}", flush=True)
    print("-" * 90, flush=True)
    for s in summary:
        fm = s["fast_iptm_median"]
        fms = s["fast_iptm_std"]
        if "full_iptm_median" in s:
            um = s["full_iptm_median"]
            ums = s["full_iptm_std"]
            delta = fm - um
            print(f"{s['name']:22s}  {fm:8.3f}  {s['fast_iptm_mean']:9.3f}  "
                  f"{fms:8.3f}  {um:8.3f}  {ums:8.3f}  {delta:+6.3f}",
                  flush=True)
        else:
            print(f"{s['name']:22s}  {fm:8.3f}  {s['fast_iptm_mean']:9.3f}  "
                  f"{fms:8.3f}  {'?':>8s}  {'?':>8s}  {'?':>6s}",
                  flush=True)

    # Rank correlation
    if all("full_iptm_median" in s for s in summary):
        from scipy.stats import spearmanr
        # By full median (Step 1 ranking)
        full_order = sorted(summary,
                            key=lambda r: r["full_iptm_median"], reverse=True)
        full_ranks = {s["name"]: i for i, s in enumerate(full_order)}
        # By fast median (Step 3 ranking)
        fast_order = sorted(summary,
                            key=lambda r: r["fast_iptm_median"], reverse=True)
        fast_ranks = {s["name"]: i for i, s in enumerate(fast_order)}
        ks = [full_ranks[s["name"]] for s in summary]
        ms = [fast_ranks[s["name"]] for s in summary]
        rho, p = spearmanr(ks, ms)
        print(f"\n  Spearman rank corr (Full median vs Fast median): "
              f"ρ={rho:.3f}  p={p:.3f}", flush=True)
        if rho > 0.7:
            print("  → Rankings AGREE: top candidates are real binding "
                  "configurations across both models", flush=True)
        elif rho > 0.4:
            print("  → Rankings PARTIALLY AGREE: noise in Fast model "
                  "reshuffles some middle candidates", flush=True)
        else:
            print("  → Rankings DISAGREE: candidate rankings are model-"
                  "specific; treat any single-model ranking as suspect",
                  flush=True)
    else:
        rho, p = None, None
        print("\n  (Step 1 results not found, skipping rank correlation)",
              flush=True)

    out = {
        "n_seeds": N_SEEDS,
        "model": "ESMFold2-Fast",
        "candidates": [{"name": n, "cdr": c, "fw_muts": m}
                       for n, c, m in TOP_CANDIDATES],
        "fast_results": fast_results,
        "summary": summary,
    }
    if rho is not None:
        out["spearman_full_vs_fast"] = float(rho)
        out["spearman_p"] = float(p)
    with open(out_dir / "crossval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/crossval.json", flush=True)


if __name__ == "__main__":
    main()
