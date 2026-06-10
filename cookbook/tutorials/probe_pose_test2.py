"""Test 2: Same framework, different CDRs — does pose differ?

v9_init (V2_STEP050):    CDR = GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR
v9_best_15seed:          CDR = GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR

Same 95-residue framework, only the 32 CDR positions differ.
If pose is framework-determined:    binder RMSD < 1Å after target-align
If pose is CDR-driven (epitope contact):  binder RMSD > 3Å
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
from design_b5_mps_v9_cacoord import extract_ca_per_token  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"
POST_H3 = "WGQGTLVTVS"


def make_full_seq(cdr32: str) -> str:
    assert len(cdr32) == 32, f"CDR must be 32, got {len(cdr32)}"
    return (PRE_H1 + cdr32[:10] + POST_H1_PRE_H2 + cdr32[10:16]
            + POST_H2_PRE_H3 + cdr32[16:] + POST_H3)


# The two sequences to compare
SEQ_INIT = make_full_seq("GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR")
SEQ_BEST = make_full_seq("GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR")

# Show diff
print(f"v9_init  CDR (32): {'GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR'}", flush=True)
print(f"v9_best  CDR (32): {'GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR'}", flush=True)
print(f"diff mask (X=changed):", flush=True)
a = "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR"
b = "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR"
diff = "".join("X" if x != y else "." for x, y in zip(a, b))
print(f"  H1  {a[:10]} vs {b[:10]}  →  {diff[:10]}", flush=True)
print(f"  H2  {a[10:16]} vs {b[10:16]}  →  {diff[10:16]}", flush=True)
print(f"  H3  {a[16:]} vs {b[16:]}  →  {diff[16:]}", flush=True)
print(f"  -> {diff.count('X')}/32 CDR positions differ", flush=True)


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


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_aligned = (Pc @ R.T) + Q.mean(0, keepdims=True)
    diff = P_aligned - Q
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def kabsch_rotation(P, Q):
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def fold_one(model, binder_seq, target_seq, seed=11):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]
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

    sample_coords = out["sample_atom_coords"].float()
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    if sample_coords.dim() == 2:
        sample_coords = sample_coords.unsqueeze(0)
    ca = extract_ca_per_token(sample_coords, atom_to_token,
                              ref_atom_name_chars, atom_mask)
    if torch.isnan(ca).any():
        ca = torch.nan_to_num(ca, nan=0.0)
    ca = ca[0].cpu().numpy()  # [L, 3], binder-first

    return {
        "iptm": iptm, "ptm": ptm, "pae": pae,
        "binder_ca": ca[:BINDER_LEN], "target_ca": ca[BINDER_LEN:],
        "time_s": dt,
    }


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    epi_in_pae = [e + BINDER_LEN for e in epi]
    print(f"\ntarget_len={len(target_seq)} binder_len={BINDER_LEN}")
    print(f"epitope (21): {epi}")
    print(f"CDR (32): {cdr[:10]} | {cdr[10:16]} | {cdr[16:]}")

    out_dir = Path("/tmp/b5_pose_test2")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()

    model = load_model_full()

    SEED = 11
    print(f"\n=== Fold v9_init (V2_STEP050) seed={SEED} ===", flush=True)
    r_init = fold_one(model, SEQ_INIT, target_seq, seed=SEED)
    print(f"  iptm={r_init['iptm']:.3f}  pTM={r_init['ptm']:.3f}  "
          f"({r_init['time_s']:.1f}s)", flush=True)
    np.save(out_dir / "init_binder_ca.npy", r_init["binder_ca"])
    np.save(out_dir / "init_target_ca.npy", r_init["target_ca"])
    np.save(out_dir / "init_pae.npy", r_init["pae"])
    pae_cdr_epi_init = r_init["pae"][np.ix_(cdr, epi_in_pae)]
    print(f"  ipSAE_min(CDR,epi)={pae_cdr_epi_init.min():.2f}  "
          f"ipSAE_p10={np.percentile(pae_cdr_epi_init, 10):.2f}  "
          f"ipSAE_mean={pae_cdr_epi_init.mean():.2f}", flush=True)

    print(f"\n=== Fold v9_best_15seed seed={SEED} ===", flush=True)
    r_best = fold_one(model, SEQ_BEST, target_seq, seed=SEED)
    print(f"  iptm={r_best['iptm']:.3f}  pTM={r_best['ptm']:.3f}  "
          f"({r_best['time_s']:.1f}s)", flush=True)
    np.save(out_dir / "best_binder_ca.npy", r_best["binder_ca"])
    np.save(out_dir / "best_target_ca.npy", r_best["target_ca"])
    np.save(out_dir / "best_pae.npy", r_best["pae"])
    pae_cdr_epi_best = r_best["pae"][np.ix_(cdr, epi_in_pae)]
    print(f"  ipSAE_min(CDR,epi)={pae_cdr_epi_best.min():.2f}  "
          f"ipSAE_p10={np.percentile(pae_cdr_epi_best, 10):.2f}  "
          f"ipSAE_mean={pae_cdr_epi_best.mean():.2f}", flush=True)

    # === Compare ===
    print(f"\n{'='*60}", flush=True)
    print(f"=== POSE COMPARISON (same framework, different CDR) ===",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    b_init = r_init["binder_ca"]
    t_init = r_init["target_ca"]
    b_best = r_best["binder_ca"]
    t_best = r_best["target_ca"]

    # Method 1: align on target (Kabsch), compare binder CAs
    offset = t_best.mean(0) - t_init.mean(0)
    b_init_aligned = b_init + offset
    r_full = kabsch_rmsd(b_init_aligned, b_best)
    print(f"  Method A (target-mean align → binder RMSD):", flush=True)
    print(f"    binder CA RMSD (v9_init → v9_best): {r_full:.2f}Å",
          flush=True)

    # Method 2: Kabsch rotation+translation (rigid align on target)
    R = kabsch_rotation(t_init, t_best)
    b_init_kab = (b_init - t_init.mean(0)) @ R.T + t_best.mean(0)
    t_init_kab = (t_init - t_init.mean(0)) @ R.T + t_best.mean(0)
    r_full2 = kabsch_rmsd(b_init_kab, b_best)
    print(f"  Method B (Kabsch align on target → binder RMSD):", flush=True)
    print(f"    binder CA RMSD (v9_init → v9_best): {r_full2:.2f}Å",
          flush=True)

    # Per-region RMSD
    print(f"\n  Per-region RMSD (Method B alignment):", flush=True)
    cdr_set = set(cdr)
    fw_idx = [i for i in range(BINDER_LEN) if i not in cdr_set]
    cdr_idx = cdr
    rmsd_fw = np.sqrt(((b_init_kab[fw_idx] - b_best[fw_idx]) ** 2).sum(-1).mean())
    rmsd_cdr = np.sqrt(((b_init_kab[cdr_idx] - b_best[cdr_idx]) ** 2).sum(-1).mean())
    rmsd_h1 = np.sqrt(((b_init_kab[cdr[:10]] - b_best[cdr[:10]]) ** 2).sum(-1).mean())
    rmsd_h2 = np.sqrt(((b_init_kab[cdr[10:16]] - b_best[cdr[10:16]]) ** 2).sum(-1).mean())
    rmsd_h3 = np.sqrt(((b_init_kab[cdr[16:]] - b_best[cdr[16:]]) ** 2).sum(-1).mean())
    print(f"    framework (95 res): {rmsd_fw:.2f}Å", flush=True)
    print(f"    CDR  total (32 res): {rmsd_cdr:.2f}Å", flush=True)
    print(f"    H1   (10 res): {rmsd_h1:.2f}Å", flush=True)
    print(f"    H2   ( 6 res): {rmsd_h2:.2f}Å", flush=True)
    print(f"    H3   (16 res): {rmsd_h3:.2f}Å", flush=True)

    # Interface centroid distance (key pose descriptor)
    init_cdr_ca = b_init_kab[cdr].mean(0)
    best_cdr_ca = b_best[cdr].mean(0)
    epi_ca_init = t_init_kab[epi].mean(0)
    epi_ca_best = t_best[epi].mean(0)
    print(f"\n  Interface centroid (CDR-mean to epitope-mean):",
          flush=True)
    print(f"    v9_init  : {np.linalg.norm(init_cdr_ca - epi_ca_init):.2f}Å",
          flush=True)
    print(f"    v9_best  : {np.linalg.norm(best_cdr_ca - epi_ca_best):.2f}Å",
          flush=True)

    # Save
    summary = {
        "seed": SEED,
        "init_cdr": "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR",
        "best_cdr": "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR",
        "n_cdr_diffs": diff.count("X"),
        "init_iptm": r_init["iptm"], "init_ptm": r_init["ptm"],
        "best_iptm": r_best["iptm"], "best_ptm": r_best["ptm"],
        "init_ipsae_min_cdr_epi": float(pae_cdr_epi_init.min()),
        "best_ipsae_min_cdr_epi": float(pae_cdr_epi_best.min()),
        "binder_rmsd_methodA": float(r_full),
        "binder_rmsd_methodB": float(r_full2),
        "framework_rmsd": float(rmsd_fw),
        "cdr_rmsd": float(rmsd_cdr),
        "h1_rmsd": float(rmsd_h1), "h2_rmsd": float(rmsd_h2),
        "h3_rmsd": float(rmsd_h3),
    }
    with open(out_dir / "test2.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Verdict:", flush=True)
    if r_full2 < 1.0:
        print(f"    RMSD {r_full2:.2f}Å < 1.0Å → pose is FRAMEWORK-determined "
              f"(CDRs interchangeable in pose space)", flush=True)
    elif r_full2 > 3.0:
        print(f"    RMSD {r_full2:.2f}Å > 3.0Å → pose is CDR-driven "
              f"(CDRs select a different basin)", flush=True)
    else:
        print(f"    RMSD {r_full2:.2f}Å ∈ [1, 3]Å → ambiguous; "
              f"CDRs modulate pose but framework sets the rough location",
              flush=True)

    print(f"\nResults saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
