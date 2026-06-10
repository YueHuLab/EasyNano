"""Step 2 of plan: Framework micro-tuning — can 1-3 framework residue changes
improve the iptm median over the top Step 1 candidates?

Step 1 showed two distinct winners:
  - v9_best_15seed: median 0.658 but bimodal (seeds 0.43 / 0.70)
  - v16_s5_s44:     median 0.551, very stable (std 0.016)
  - v16_init_v2s050: median 0.541, most stable (std 0.013)

Hypothesis from Test 2: framework shift is 0.29Å, framework is rigid.
But: framework determines WHICH pose basin you land in. Vernier-zone
mutations (residues that pack against CDRs) could shift the basin and
stabilize/improve iptm.

Strategy:
  1. Take TOP-2 Step 1 candidates as parents (v9_best_15seed, v16_s5_s44).
  2. Try ~10 single-residue framework mutations at vernier positions.
  3. N=3 seeds per mutation.
  4. Compute median iptm and ipsAE_p10.
  5. Compare to baseline (no mutation).

If any mutation beats baseline median iptm by ≥0.02, it's a candidate for
Step 3 (cross-model validation).
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
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_b5_pdb import setup_design  # noqa: E402
from design_b5_mps_v9_cacoord import extract_ca_per_token  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127
N_SEEDS = 3

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"
POST_H3 = "WGQGTLVTVS"

# The full sequence template (1-indexed by binder position, 0-indexed by seq)
def make_full_seq(cdr32: str, fw_mutations: List[Tuple[int, str]] = None) -> str:
    """cdr32 is 32 chars. fw_mutations is list of (0-indexed binder pos, new_aa)
    where pos is 0..126 and pos not in CDR set (we don't validate here)."""
    seq = (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
           + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)
    assert len(seq) == 127, f"len {len(seq)}"
    if fw_mutations:
        seq_list = list(seq)
        for pos, new_aa in fw_mutations:
            assert 0 <= pos < 127
            assert len(new_aa) == 1
            old = seq_list[pos]
            seq_list[pos] = new_aa
            print(f"    mutation: pos {pos}  {old} → {new_aa}", flush=True)
        seq = "".join(seq_list)
    return seq


# Framework template (for reference)
FW_TEMPLATE = PRE_H1 + "##########" + POST_H1_PRE_H2 + "######" + POST_H2_PRE_H3 + "################" + POST_H3
# Identify 0-indexed framework positions adjacent to CDRs:
CDR_SET = set(range(25, 35)) | set(range(54, 60)) | set(range(101, 117))
# Vernier / flank positions to test:
#   - 23 (last fw before H1, 0-indexed = 23, 1-indexed = 24)
#   - 34 (last fw of H1 N-region, 0-indexed = 24)
# Actually, our CDR list is 1-indexed in the prompt but Python is 0-indexed.
# Let me re-derive from make_full_seq:
#   pos 0..23  = PRE_H1 (24 chars)
#   pos 24..33 = CDR H1 (10 chars)
#   pos 34..52 = POST_H1_PRE_H2 (19 chars)
#   pos 53..58 = CDR H2 (6 chars)
#   pos 59..99 = POST_H2_PRE_H3 (41 chars)
#   pos 100..115 = CDR H3 (16 chars)
#   pos 116..126 = POST_H3 (11 chars)
# Total = 24+10+19+6+41+16+11 = 127 ✓
# So 0-indexed framework positions adjacent to CDRs:
#   - 23 (last of PRE_H1, just before H1)
#   - 34 (first of POST_H1_PRE_H2, just after H1)
#   - 52 (last of POST_H1_PRE_H2, just before H2)
#   - 59 (first of POST_H2_PRE_H3, just after H2)
#   - 99 (last of POST_H2_PRE_H3, just before H3)  -- H3 N-flank is critical
#   - 116 (first of POST_H3, just after H3)
# These are the 6 "flank" positions. Add a few vernier:
#   - 12 (vernier zone 1, in PRE_H1 mid)
#   - 71 (vernier zone 2, in POST_H2_PRE_H3 mid)

# Try a small set of conservative mutations
MUTATIONS_TO_TRY = [
    # (label, [(0-indexed pos, new_aa), ...])
    ("baseline",        []),
    ("p23S→A",          [(23, "A")]),  # H1 N-flank vernier
    ("p34S→A",          [(34, "A")]),  # H1 C-flank vernier
    ("p52I→V",          [(52, "V")]),  # H2 N-flank
    ("p59T→A",          [(59, "A")]),  # H2 C-flank
    ("p99Y→F",          [(99, "F")]),  # H3 N-flank (H3 orientation)
    ("p99Y→W",          [(99, "W")]),
    ("p116W→F",         [(116, "F")]), # H3 C-flank
    ("p116W→Y",         [(116, "Y")]),
    ("p12G→A",          [(12, "A")]),  # framework mid (vernier)
    ("p71L→V",          [(71, "V")]),  # framework mid (vernier)
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
    epi_in_pae = np.array([e + BINDER_LEN for e in epi])
    cdr_arr = np.array(cdr)
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}", flush=True)
    print(f"epitope (21): {epi}", flush=True)
    print(f"CDR (32): {cdr[:10]} | {cdr[10:16]} | {cdr[16:]}", flush=True)

    out_dir = Path("/tmp/b5_finetune")
    out_dir.mkdir(exist_ok=True)

    # Pick top-2 parents from Step 1
    PARENT_CDRS = []  # list of (name, cdr)
    step1_path = Path("/tmp/b5_multiseed/multiseed.json")
    if step1_path.exists():
        with open(step1_path) as f:
            step1 = json.load(f)
        # Top-2 by median: best + most stable
        cand_by_name = {c["name"]: c for c in step1["candidates"]}
        sorted_by_med = step1["summary_sorted_by_median"]
        for entry in sorted_by_med[:2]:
            c = cand_by_name[entry["name"]]
            PARENT_CDRS.append((entry["name"], c["cdr"]))
        print(f"\nStep 1 parents (top-2 by median):", flush=True)
        for name, cdr in PARENT_CDRS:
            print(f"  {name}  CDR={cdr}", flush=True)
    else:
        # Fallback
        PARENT_CDRS = [
            ("v9_best_15seed_fallback", "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR"),
            ("v16_s5_s44_fallback",     "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR"),
        ]
        print(f"\nNo Step 1 results, using fallbacks: {PARENT_CDRS}", flush=True)

    model = load_model_full()
    seeds = list(range(N_SEEDS))

    print(f"\n{'='*70}", flush=True)
    print(f"=== Step 2: Framework micro-tuning ===", flush=True)
    print(f"  Parents: {len(PARENT_CDRS)}", flush=True)
    print(f"  Mutations per parent: {len(MUTATIONS_TO_TRY)}", flush=True)
    print(f"  Seeds per (parent, mutation): {N_SEEDS}", flush=True)
    print(f"  Total folds: {len(PARENT_CDRS) * len(MUTATIONS_TO_TRY) * N_SEEDS}",
          flush=True)
    print(f"{'='*70}\n", flush=True)

    all_results = {}  # parent_name -> {label -> results}
    for parent_name, parent_cdr in PARENT_CDRS:
        print(f"\n{'#'*70}", flush=True)
        print(f"### Parent: {parent_name}  (CDR = {parent_cdr}) ###", flush=True)
        print(f"{'#'*70}\n", flush=True)
        all_results[parent_name] = {"cdr": parent_cdr, "variants": {}}
        for label, muts in MUTATIONS_TO_TRY:
            print(f"--- {label} ---", flush=True)
            full_seq = make_full_seq(parent_cdr, fw_mutations=muts)
            all_results[parent_name]["variants"][label] = {
                "muts": muts, "iptm": {}, "ptm": {},
                "ipsae_p10": {}, "iptm_arr": []}
            for s in seeds:
                r = fold_one(model, full_seq, target_seq, seed=s)
                pae_cdr_epi = r["pae"][np.ix_(cdr_arr, epi_in_pae)]
                ipsae_p10v = float(np.percentile(pae_cdr_epi, 10))
                all_results[parent_name]["variants"][label]["iptm"][s] = r["iptm"]
                all_results[parent_name]["variants"][label]["ptm"][s] = r["ptm"]
                all_results[parent_name]["variants"][label]["ipsae_p10"][s] = ipsae_p10v
                all_results[parent_name]["variants"][label]["iptm_arr"].append(r["iptm"])
                print(f"  seed={s}  iptm={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
                      f"ipSAE_p10={ipsae_p10v:.2f}  ({r['time_s']:.1f}s)",
                      flush=True)
            iptms = all_results[parent_name]["variants"][label]["iptm_arr"]
            ips = list(all_results[parent_name]["variants"][label]["ipsae_p10"].values())
            print(f"  → iptm: median={np.median(iptms):.3f}  "
                  f"mean={np.mean(iptms):.3f}  std={np.std(iptms):.3f}  "
                  f"min={min(iptms):.3f}  max={max(iptms):.3f}", flush=True)
            print(f"  → ipSAE_p10: median={np.median(ips):.2f}", flush=True)
            print(flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("=== SUMMARY (sorted by median iptm) ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    summary = []
    for parent_name, _ in PARENT_CDRS:
        for label, muts in MUTATIONS_TO_TRY:
            v = all_results[parent_name]["variants"][label]
            iptms = v["iptm_arr"]
            ips = list(v["ipsae_p10"].values())
            summary.append({
                "parent": parent_name, "label": label, "muts": muts,
                "iptm_median": float(np.median(iptms)),
                "iptm_mean": float(np.mean(iptms)),
                "iptm_std": float(np.std(iptms)),
                "iptm_min": float(min(iptms)),
                "iptm_max": float(max(iptms)),
                "ipsae_p10_median": float(np.median(ips)),
            })

    summary.sort(key=lambda r: r["iptm_median"], reverse=True)
    print(f"{'parent':22s}  {'label':14s}  {'med':>5s}  {'mean':>5s}  "
          f"{'std':>5s}  {'min':>5s}  {'max':>5s}  {'ipSAE_p10':>8s}  "
          f"muts", flush=True)
    print("-" * 110, flush=True)
    for s in summary:
        muts_str = ",".join(f"{p}→{new}" for p, new in s["muts"])
        if not muts_str:
            muts_str = "(baseline)"
        print(f"{s['parent']:22s}  {s['label']:14s}  {s['iptm_median']:5.3f}  "
              f"{s['iptm_mean']:5.3f}  {s['iptm_std']:5.3f}  "
              f"{s['iptm_min']:5.3f}  {s['iptm_max']:5.3f}  "
              f"{s['ipsae_p10_median']:8.2f}  {muts_str}", flush=True)

    # Per-parent delta from baseline
    for parent_name, _ in PARENT_CDRS:
        base = next(s for s in summary
                    if s["parent"] == parent_name and s["label"] == "baseline")
        base_med = base["iptm_median"]
        print(f"\n  --- {parent_name} Δ from baseline (med={base_med:.3f}) ---",
              flush=True)
        for s in [x for x in summary if x["parent"] == parent_name]:
            if s["label"] == "baseline":
                continue
            delta = s["iptm_median"] - base_med
            marker = ""
            if delta > 0.02:
                marker = " ← IMPROVED"
            elif delta < -0.05:
                marker = " ← REGRESSED"
            muts_str = ",".join(f"{p}→{new}" for p, new in s["muts"])
            print(f"    {s['label']:14s}  Δ_med={delta:+.3f}  "
                  f"med={s['iptm_median']:.3f}  std={s['iptm_std']:.3f}  "
                  f"{muts_str}{marker}", flush=True)

    out = {
        "n_seeds": N_SEEDS,
        "parent_cdrs": PARENT_CDRS,
        "mutations_tried": MUTATIONS_TO_TRY,
        "all_results": all_results,
        "summary_sorted_by_median": summary,
    }
    with open(out_dir / "finetune.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}/finetune.json", flush=True)


if __name__ == "__main__":
    main()
