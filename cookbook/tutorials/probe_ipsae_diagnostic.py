"""ipSAE diagnostic on known B5 sequences.

Hypothesis:  ipSAE_min(CDR, epitope) is a sharper binding signal than ipTM
              (it ignores non-interface residue pairs).

Test:        Fold 7 sequences spanning iptm 0.10..0.72, compute ipSAE,
              report correlation with the known iptm.

Output:      /tmp/b5_ipsae_diag/
              - <name>_pae.npy      full [350, 350] PAE matrix
              - results.json        all metrics
              - pae_cdr_epi.txt     <32, 21> PAE submatrix (CDR x epitope)
                                    for each sequence (ASCII)
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import math
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_b5_pdb import setup_design  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"  # 25
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"  # 19  (starts with S, not M)
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"  # 41  (starts with T, not ST)
POST_H3 = "WGQGTLVTVS"  # 10


def make_full_seq(cdr32: str) -> str:
    assert len(cdr32) == 32, f"CDR must be 32, got {len(cdr32)}"
    return (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
            + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)


# name, CDR (32), known_iptm, source/notes
SEQUENCES = [
    ("v9_best_15seed",  "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR", 0.717,
     "v9 15-seed global best (B5_V15_PERIODIC_REANCHOR.md)"),
    ("v9_step48",       "GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR", 0.619,
     "v9 step 48 baseline (in v16 multistart eval)"),
    ("v16_s5_s44",      "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR", 0.572,
     "v16 best, seed5 step44"),
    ("v16_s5_s56",      "GLQIGYGNYMSYSGQKRVVTDSSQPIYKAPQR", 0.539,
     "v16 #2, seed5 step56"),
    ("v16_s2_s44",      "GLQIGYGWYMSYSGQSRVVTDSSTPIYKAPIR", 0.527,
     "v16 #3, seed2 step44"),
    ("v16_init_v2s050", "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR", 0.471,
     "v16 init (V2_STEP050_FULL)"),
    ("v16_s4_s44",      "GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPIR", 0.104,
     "v16 worst, seed4 step44"),
]


def load_model_full():
    print("Loading FULL ESMFold2 (1.3G) ...", flush=True)
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


def fold_one(model, binder_seq, target_seq):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=3, num_sampling_steps=14,
            num_diffusion_samples=1, calculate_confidence=True,
        )
    return out


def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx * sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (sx * sy)


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    target_len = len(target_seq)
    print(f"target_len={target_len} binder_len={BINDER_LEN}", flush=True)
    print(f"epitope (21, target-coord): {epi}", flush=True)
    print(f"CDR (32, binder-coord): {cdr[:10]} | {cdr[10:16]} | {cdr[16:]}",
          flush=True)

    epi_in_pae = [e + BINDER_LEN for e in epi]
    print(f"epitope in pae coord: {epi_in_pae}", flush=True)

    out_dir = Path("/tmp/b5_ipsae_diag")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()

    model = load_model_full()

    results = []
    for name, cdr32, known_iptm, source in SEQUENCES:
        binder_seq = make_full_seq(cdr32)
        assert len(binder_seq) == 127, f"binder len {len(binder_seq)}"
        print(f"\n--- {name}  (known iptm={known_iptm}) ---", flush=True)
        print(f"  binder[25:117]={binder_seq[25:117]}", flush=True)
        t0 = time.time()
        out = fold_one(model, binder_seq, target_seq)
        dt = time.time() - t0

        iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
        ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
        pae = out["pae"][0].float().cpu().numpy()  # [L, L]
        cpi = out["pair_chains_iptm"][0].cpu().numpy()  # [2, 2]

        pae_bt = pae[:BINDER_LEN, BINDER_LEN:]            # [127, 223]
        pae_cdr_epi = pae[np.ix_(cdr, epi_in_pae)]         # [32, 21]

        m = {
            "name": name, "source": source, "known_iptm": known_iptm,
            "fold_iptm": iptm, "fold_ptm": ptm, "fold_time_s": dt,
            "ipsae_min_cdr_epi": float(pae_cdr_epi.min()),
            "ipsae_mean_cdr_epi": float(pae_cdr_epi.mean()),
            "ipsae_p10_cdr_epi": float(np.percentile(pae_cdr_epi, 10)),
            "ipsae_min_bt": float(pae_bt.min()),
            "ipsae_mean_bt": float(pae_bt.mean()),
            "cp_iptm_bt": float(cpi[0, 1]),
            "cp_iptm_bb": float(cpi[0, 0]),
            "cp_iptm_tt": float(cpi[1, 1]),
        }
        results.append(m)
        np.save(out_dir / f"{name}_pae.npy", pae)
        np.save(out_dir / f"{name}_pae_cdr_epi.npy", pae_cdr_epi)

        print(f"  iptm={iptm:.3f}  pTM={ptm:.3f}  ({dt:.1f}s)", flush=True)
        print(f"  ipSAE_min(CDR,epi)  = {m['ipsae_min_cdr_epi']:>6.2f}Å",
              flush=True)
        print(f"  ipSAE_p10(CDR,epi)  = {m['ipsae_p10_cdr_epi']:>6.2f}Å",
              flush=True)
        print(f"  ipSAE_mean(CDR,epi) = {m['ipsae_mean_cdr_epi']:>6.2f}Å",
              flush=True)
        print(f"  ipSAE_min(b,t)      = {m['ipsae_min_bt']:>6.2f}Å",
              flush=True)
        print(f"  ipSAE_mean(b,t)     = {m['ipsae_mean_bt']:>6.2f}Å",
              flush=True)
        print(f"  chain_pair_iptm(b,t)= {m['cp_iptm_bt']:.3f}", flush=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n========== SUMMARY (sorted by known iptm desc) ==========",
          flush=True)
    print(f"  {'name':>20}  {'known':>6}  {'fold':>6}  "
          f"{'min(c,e)':>8}  {'p10(c,e)':>8}  {'mean(c,e)':>9}  "
          f"{'min(b,t)':>8}  {'mean(b,t)':>9}  {'cp_bt':>6}",
          flush=True)
    print(f"  {'-'*92}", flush=True)
    for r in sorted(results, key=lambda r: -r["known_iptm"]):
        print(f"  {r['name']:>20}  {r['known_iptm']:>6.3f}  "
              f"{r['fold_iptm']:>6.3f}  "
              f"{r['ipsae_min_cdr_epi']:>8.2f}  "
              f"{r['ipsae_p10_cdr_epi']:>8.2f}  "
              f"{r['ipsae_mean_cdr_epi']:>9.2f}  "
              f"{r['ipsae_min_bt']:>8.2f}  "
              f"{r['ipsae_mean_bt']:>9.2f}  "
              f"{r['cp_iptm_bt']:>6.3f}", flush=True)

    k = [r["known_iptm"] for r in results]
    metrics = {
        "min(cdr,epi)": [r["ipsae_min_cdr_epi"] for r in results],
        "p10(cdr,epi)": [r["ipsae_p10_cdr_epi"] for r in results],
        "mean(cdr,epi)": [r["ipsae_mean_cdr_epi"] for r in results],
        "min(b,t)": [r["ipsae_min_bt"] for r in results],
        "mean(b,t)": [r["ipsae_mean_bt"] for r in results],
        "cp_iptm(b,t)": [r["cp_iptm_bt"] for r in results],
        "fold_iptm": [r["fold_iptm"] for r in results],
    }
    print("\n--- Pearson correlation with known iptm ---", flush=True)
    for name, vals in metrics.items():
        c = pearson(k, vals)
        # Negative correlation is "good" for ipSAE metrics (lower PAE = better)
        sign = " (LOWER=better)" if "ipSAE" in name or "min(" in name or "mean(" in name else ""
        print(f"  {name:>16}: r = {c:+.3f}{sign}", flush=True)

    # ASCII heatmap of PAE(CDR, epi) for top-2 and bottom-1 sequences
    print("\n--- PAE[CDR, epi] submatrix (32x21, in Angstroms) ---",
          flush=True)
    for r in sorted(results, key=lambda r: -r["known_iptm"])[:2] + \
             sorted(results, key=lambda r: r["known_iptm"])[:1]:
        arr = np.load(out_dir / f"{r['name']}_pae_cdr_epi.npy")
        print(f"\n  {r['name']}  known_iptm={r['known_iptm']}", flush=True)
        print("   CDR(each row, 32 aa) \\ epi(21 cols)", flush=True)
        # Column header (epi indices)
        hdr = "      " + " ".join(f"{e:>3d}" for e in epi)
        print(hdr, flush=True)
        for i, cdr_idx in enumerate(cdr):
            row_label = ("H1" if i < 10 else "H2" if i < 16 else "H3")
            row = " ".join(f"{arr[i, j]:>3.0f}" for j in range(21))
            print(f"  {row_label} {cdr_idx:>3d}  {row}", flush=True)

    print(f"\nResults saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
