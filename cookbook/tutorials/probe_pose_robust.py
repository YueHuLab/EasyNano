"""Step 3 of plan: Within-model robustness check with multi-sample averaging.

Original plan called for cross-model validation (Boltz-1 / AF2) but the
environment lacks network access to download Boltz's CCD dictionary
(timeout) and no AF2 weights are present. ESMFold2-Fast gives iptm ≈ 0.06
for all sequences (its confidence head isn't calibrated for inference).

Best available alternative: a "deep noise average" check using the Full
ESMFold2 with num_diffusion_samples=3 and 3 seeds. The pre-existing
Steps 1-2 used num_diffusion_samples=1 + 3-5 seeds; here we go to
3 samples × 3 seeds = 9 effective samples per sequence. This catches
candidates that are robustly high vs lucky-once.

Candidates tested: top-3 from Step 1 + the Step 2 winner
(v9_best_15seed + p116W→Y).
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
N_DIFF_SAMPLES = 3

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


# Top-3 from Step 1 + Step 2 winner
CANDIDATES = [
    ("v9_best_15seed",       "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR",  []),
    ("v9_best_15seed_p116Y", "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR",  [(116, "Y")]),
    ("v16_s5_s56",           "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR",  []),
    ("v16_s5_s56_p71V",      "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR",  [(71, "V")]),
    ("v16_s5_s44",           "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR",  []),
]


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
    # iptm / ptm are per-sample when num_diffusion_samples > 1
    iptm = out["iptm"].float().cpu().numpy()
    ptm = out["ptm"].float().cpu().numpy()
    pae = out["pae"][0].float().cpu().numpy()  # [L, L]
    return {
        "iptm_per_sample": iptm.tolist(),
        "ptm_per_sample": ptm.tolist(),
        "iptm_mean": float(np.mean(iptm)),
        "iptm_max": float(np.max(iptm)),
        "ptm_mean": float(np.mean(ptm)),
        "pae": pae,
        "time_s": dt,
    }


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    epi_in_pae = np.array([e + BINDER_LEN for e in epi])
    cdr_arr = np.array(cdr)
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}", flush=True)

    out_dir = Path("/tmp/b5_robust")
    out_dir.mkdir(exist_ok=True)

    # Reference: Step 1 single-seed-per-call iptm
    step1_path = Path("/tmp/b5_multiseed/multiseed.json")
    step1_summary = {}
    if step1_path.exists():
        with open(step1_path) as f:
            step1 = json.load(f)
        for c in step1["candidates"]:
            step1_summary[c["name"]] = c

    # Reference: Step 2 micro-tuning
    step2_path = Path("/tmp/b5_finetune/finetune.json")
    step2_summary = {}
    if step2_path.exists():
        with open(step2_path) as f:
            step2 = json.load(f)
        for s in step2.get("summary_sorted_by_median", []):
            step2_summary[s["parent"] + "_" + s["label"]] = s

    seeds = list(range(N_SEEDS))
    robust_results = {}

    print(f"\n{'='*70}", flush=True)
    print(f"=== Step 3 (alt): Within-model robustness check ===", flush=True)
    print(f"  Model: Full ESMFold2 with num_diffusion_samples={N_DIFF_SAMPLES}",
          flush=True)
    print(f"  Seeds: {N_SEEDS}", flush=True)
    print(f"  Effective samples per sequence: "
          f"{N_SEEDS * N_DIFF_SAMPLES}", flush=True)
    print(f"  Candidates: {len(CANDIDATES)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    model = load_model()

    for name, cdr_seq, fw_muts in CANDIDATES:
        full_seq = make_full_seq(cdr_seq, fw_muts=fw_muts)
        robust_results[name] = {"fw_muts": fw_muts,
                                "iptm_per_call": [],
                                "iptm_all_samples": [],
                                "ipsae_p10": []}
        print(f"--- {name} ---", flush=True)
        if fw_muts:
            for p, a in fw_muts:
                print(f"  Framework mutation: pos{p} → {a}", flush=True)
        if name in step1_summary:
            s1 = step1_summary[name]
            print(f"  Step 1 (Full, single-sample):  "
                  f"median={s1.get('iptm_median', '?')}  "
                  f"std={s1.get('iptm_std', '?')}", flush=True)
        for s in seeds:
            r = fold_one(model, full_seq, target_seq, seed=s,
                         n_diff_samples=N_DIFF_SAMPLES)
            # ipsae_p10 uses the pae (from sample 0)
            pae_cdr_epi = r["pae"][np.ix_(cdr_arr, epi_in_pae)]
            ipsae_p10v = float(np.percentile(pae_cdr_epi, 10))
            # For multi-sample: take per-sample iptm
            robust_results[name]["iptm_per_call"].append(r["iptm_mean"])
            robust_results[name]["iptm_all_samples"].extend(
                r["iptm_per_sample"])
            robust_results[name]["ipsae_p10"].append(ipsae_p10v)
            print(f"  seed={s}  iptm_per_sample={[f'{x:.3f}' for x in r['iptm_per_sample']]}  "
                  f"mean={r['iptm_mean']:.3f}  max={r['iptm_max']:.3f}  "
                  f"ipSAE_p10={ipsae_p10v:.2f}  ({r['time_s']:.1f}s)",
                  flush=True)
        all_samples = robust_results[name]["iptm_all_samples"]
        call_means = robust_results[name]["iptm_per_call"]
        print(f"  → all samples (n={len(all_samples)}):  "
              f"median={np.median(all_samples):.3f}  "
              f"mean={np.mean(all_samples):.3f}  std={np.std(all_samples):.3f}  "
              f"min={min(all_samples):.3f}  max={max(all_samples):.3f}", flush=True)
        print(flush=True)

    # === Final summary across all 3 steps ===
    print(f"\n{'='*70}", flush=True)
    print("=== STEP 3 SUMMARY (robust multi-sample) ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    summary = []
    for name, _, _ in CANDIDATES:
        all_samples = robust_results[name]["iptm_all_samples"]
        summary.append({
            "name": name,
            "robust_n": len(all_samples),
            "robust_median": float(np.median(all_samples)),
            "robust_mean": float(np.mean(all_samples)),
            "robust_std": float(np.std(all_samples)),
            "robust_min": float(min(all_samples)),
            "robust_max": float(max(all_samples)),
        })
        # Match to Step 1
        if name in step1_summary:
            s1 = step1_summary[name]
            summary[-1]["step1_median"] = s1.get("iptm_median")
            summary[-1]["step1_std"] = s1.get("iptm_std")

    summary.sort(key=lambda r: r["robust_median"], reverse=True)
    print(f"{'name':28s}  {'robust_n':>8s}  {'med':>5s}  {'mean':>5s}  "
          f"{'std':>5s}  {'min':>5s}  {'max':>5s}  "
          f"{'step1_med':>9s}  {'step1_std':>9s}", flush=True)
    print("-" * 100, flush=True)
    for s in summary:
        s1m_raw = s.get("step1_median")
        s1s_raw = s.get("step1_std")
        s1m = f"{s1m_raw:.3f}" if isinstance(s1m_raw, (int, float)) else "?"
        s1s = f"{s1s_raw:.3f}" if isinstance(s1s_raw, (int, float)) else "?"
        print(f"{s['name']:28s}  {s['robust_n']:8d}  {s['robust_median']:5.3f}  "
              f"{s['robust_mean']:5.3f}  {s['robust_std']:5.3f}  "
              f"{s['robust_min']:5.3f}  {s['robust_max']:5.3f}  "
              f"{s1m:>9s}  {s1s:>9s}", flush=True)

    # Rank agreement: do the top-3 by robust agree with Step 1?
    top3_robust = [s["name"] for s in summary[:3]]
    if step1_summary:
        step1_sorted = sorted(step1_summary.values(),
                              key=lambda c: c.get("iptm_median", 0),
                              reverse=True)
        top3_step1 = [c["name"] for c in step1_sorted[:3]]
        overlap = len(set(top3_robust) & set(top3_step1))
        print(f"\n  Top-3 agreement (robust vs Step 1): {overlap}/3",
              flush=True)
        print(f"    robust top-3: {top3_robust}", flush=True)
        print(f"    step1  top-3: {top3_step1}", flush=True)
        if overlap >= 2:
            print("  → Top candidates are ROBUST to multi-sample averaging:",
                  flush=True)
            print("    the same sequences rise to the top.", flush=True)
        else:
            print("  → Top candidates DIFFER between single-sample and",
                  flush=True)
            print("    multi-sample: single-seed iptm is noisy.", flush=True)

    out = {
        "n_seeds": N_SEEDS,
        "n_diff_samples": N_DIFF_SAMPLES,
        "model": "Full ESMFold2 (num_diffusion_samples=N_DIFF_SAMPLES)",
        "candidates": [{"name": n, "cdr": c, "fw_muts": m}
                       for n, c, m in CANDIDATES],
        "robust_results": robust_results,
        "summary": summary,
    }
    with open(out_dir / "robust.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/robust.json", flush=True)


if __name__ == "__main__":
    main()
