"""N=5 pose diagnostic: same sequence, 5 folds with different noise seeds.

Tests whether Full ESMFold2 collapses to a single pose basin or scatters
across multiple basins for a fixed binder sequence.

Target:  v9_best_15seed (CDR = GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR, the
         sequence that once scored iptm=0.717 in a 15-seed v9 run).
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

CDR = "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR"
BINDER = (PRE_H1 + CDR[:10] + POST_H1_PRE_H2 + CDR[10:16]
          + POST_H2_PRE_H3 + CDR[16:] + POST_H3)
assert len(BINDER) == 127

N_FOLDS = 5
SEEDS = [11, 23, 47, 89, 137]


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


def kabsch_rmsd(P, Q):
    """Kabsch: align P onto Q, return RMSD of aligned P vs Q."""
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


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    epi_in_pae = [e + BINDER_LEN for e in epi]
    target_len = len(target_seq)
    print(f"target_len={target_len} binder_len={BINDER_LEN}")
    print(f"epitope (21): {epi}")
    print(f"CDR: {CDR}")
    print(f"N folds: {N_FOLDS}, seeds: {SEEDS}\n", flush=True)

    out_dir = Path("/tmp/b5_pose_n5")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()

    model = load_model_full()

    from esmscore._complex import build_complex_features
    feats = build_complex_features(BINDER, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]

    folds = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
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
        pae_cdr_epi = pae[np.ix_(cdr, epi_in_pae)]
        ipsae_min = float(pae_cdr_epi.min())
        ipsae_p10 = float(np.percentile(pae_cdr_epi, 10))
        ipsae_mean = float(pae_cdr_epi.mean())
        ipsae_min_bt = float(pae[:BINDER_LEN, BINDER_LEN:].min())

        # CA coords
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

        binder_ca = ca[:BINDER_LEN]      # [127, 3]
        target_ca = ca[BINDER_LEN:]      # [223, 3]

        rec = {
            "seed": s, "fold_iptm": iptm, "fold_ptm": ptm,
            "fold_time_s": dt, "ipsae_min_cdr_epi": ipsae_min,
            "ipsae_p10_cdr_epi": ipsae_p10, "ipsae_mean_cdr_epi": ipsae_mean,
            "ipsae_min_bt": ipsae_min_bt,
        }
        folds.append(rec)
        np.save(out_dir / f"seed{s}_binder_ca.npy", binder_ca)
        np.save(out_dir / f"seed{s}_target_ca.npy", target_ca)
        np.save(out_dir / f"seed{s}_pae.npy", pae)
        print(f"  seed={s:>3d}  iptm={iptm:.3f}  pTM={ptm:.3f}  "
              f"ipSAE_min(c,e)={ipsae_min:>5.2f}  ipSAE_p10={ipsae_p10:>5.2f}  "
              f"({dt:.1f}s)", flush=True)

    # Pairwise binder CA RMSD: align on target, then RMSD on binder
    print(f"\n--- Pairwise binder CA RMSD (Å) after target-alignment ---",
          flush=True)
    print("  Aligning fold_b to fold_a via Kabsch on target CAs, then",
          flush=True)
    print("  computing RMSD on the 127 binder CA atoms.\n", flush=True)
    hdr = "        " + "".join(f"  s{s:>3d}    " for s in SEEDS)
    print(hdr, flush=True)
    rmsd_matrix = np.zeros((N_FOLDS, N_FOLDS))
    for i, si in enumerate(SEEDS):
        bi = np.load(out_dir / f"seed{si}_binder_ca.npy")
        ti = np.load(out_dir / f"seed{si}_target_ca.npy")
        row = f"  s{si:>3d}  "
        for j, sj in enumerate(SEEDS):
            bj = np.load(out_dir / f"seed{sj}_binder_ca.npy")
            tj = np.load(out_dir / f"seed{sj}_target_ca.npy")
            r = kabsch_rmsd(bj, bi)  # align bj's binder to bi's binder (shifted by tj-ti)
            # actually, we want to align based on target. Let me redo:
            # align bj+offset to bi (where offset comes from target alignment)
            # The translation from tj to ti is: ti.mean() - tj.mean()
            # Apply this to bj, then Kabsch on bj_translated vs bi
            offset = ti.mean(0) - tj.mean(0)
            bj_aligned = bj + offset
            r = kabsch_rmsd(bj_aligned, bi)
            rmsd_matrix[i, j] = r
            row += f"{r:>8.2f}"
        print(row, flush=True)

    # Pairwise ipSAE and iptm deltas
    print(f"\n--- Pairwise metric deltas ---", flush=True)
    print(f"  iptm range across N: "
          f"[{min(r['fold_iptm'] for r in folds):.3f}, "
          f"{max(r['fold_iptm'] for r in folds):.3f}], "
          f"median={sorted(r['fold_iptm'] for r in folds)[N_FOLDS//2]:.3f}",
          flush=True)
    print(f"  ipSAE_min(CDR,epi) range: "
          f"[{min(r['ipsae_min_cdr_epi'] for r in folds):.2f}, "
          f"{max(r['ipsae_min_cdr_epi'] for r in folds):.2f}], "
          f"median={sorted(r['ipsae_min_cdr_epi'] for r in folds)[N_FOLDS//2]:.2f}",
          flush=True)

    # RMSD summary
    triu = rmsd_matrix[np.triu_indices(N_FOLDS, k=1)]
    print(f"  Pairwise binder CA RMSD: "
          f"min={triu.min():.2f}Å  median={np.median(triu):.2f}Å  "
          f"max={triu.max():.2f}Å", flush=True)

    # Naive clustering: if min RMSD < 3Å between ANY two, call it a basin
    close_pairs = (triu < 5.0).sum()
    very_close_pairs = (triu < 3.0).sum()
    print(f"  Pairs with RMSD < 5Å: {close_pairs}/{len(triu)}", flush=True)
    print(f"  Pairs with RMSD < 3Å: {very_close_pairs}/{len(triu)}", flush=True)

    # Per-atom variance: how stable is each binder residue's position across N?
    binder_cas = np.stack([np.load(out_dir / f"seed{s}_binder_ca.npy")
                           for s in SEEDS])  # [N, 127, 3]
    # But we need to align first. Use fold 0 as reference.
    ref_target = np.load(out_dir / f"seed{SEEDS[0]}_target_ca.npy")
    aligned_cas = []
    for i, s in enumerate(SEEDS):
        ti = np.load(out_dir / f"seed{s}_target_ca.npy")
        offset = ref_target.mean(0) - ti.mean(0)
        bi = np.load(out_dir / f"seed{s}_binder_ca.npy")
        bi_aligned = bi + offset
        # Kabsch align bi_aligned to binder_cas[0]
        R = kabsch_rotation(bi_aligned, binder_cas[0])
        bi_aligned = (bi_aligned - bi_aligned.mean(0)) @ R.T + binder_cas[0].mean(0)
        aligned_cas.append(bi_aligned)
    aligned_cas = np.stack(aligned_cas)  # [N, 127, 3]
    per_res_std = aligned_cas.std(axis=0).max(axis=-1)  # [127], max std over xyz
    # CDR residues
    cdr_set = set(cdr)
    framework_std = np.array([per_res_std[i] for i in range(BINDER_LEN) if i not in cdr_set])
    cdr_std = np.array([per_res_std[i] for i in cdr])
    print(f"\n  Per-residue position std (max over xyz, across N folds):", flush=True)
    print(f"    framework (95 res): median={np.median(framework_std):.2f}Å  "
          f"max={framework_std.max():.2f}Å", flush=True)
    print(f"    CDR (32 res):       median={np.median(cdr_std):.2f}Å  "
          f"max={cdr_std.max():.2f}Å", flush=True)

    # Save
    with open(out_dir / "folds.json", "w") as f:
        json.dump({
            "binder_seq": BINDER, "cdr": CDR,
            "n_folds": N_FOLDS, "seeds": SEEDS,
            "folds": folds,
            "rmsd_matrix": rmsd_matrix.tolist(),
            "framework_std_median": float(np.median(framework_std)),
            "framework_std_max": float(framework_std.max()),
            "cdr_std_median": float(np.median(cdr_std)),
            "cdr_std_max": float(cdr_std.max()),
        }, f, indent=2)

    # Best-fold pose vs median-fold
    iptms = np.array([r["fold_iptm"] for r in folds])
    pae_mins = np.array([r["ipsae_min_cdr_epi"] for r in folds])
    best_idx = int(iptms.argmax())
    best_pae_idx = int(pae_mins.argmin())
    print(f"\n  Best fold by iptm:   seed={SEEDS[best_idx]}  "
          f"iptm={iptms[best_idx]:.3f}  ipSAE={pae_mins[best_idx]:.2f}", flush=True)
    print(f"  Best fold by ipSAE:  seed={SEEDS[best_pae_idx]}  "
          f"iptm={iptms[best_pae_idx]:.3f}  ipSAE={pae_mins[best_pae_idx]:.2f}",
          flush=True)
    print(f"  RMSD(best_iptm, best_ipSAE) = "
          f"{rmsd_matrix[best_idx, best_pae_idx]:.2f}Å", flush=True)
    print(f"\nResults saved to {out_dir}/", flush=True)


def kabsch_rotation(P, Q):
    """Return rotation matrix that aligns P (centred) onto Q (centred)."""
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


if __name__ == "__main__":
    main()
