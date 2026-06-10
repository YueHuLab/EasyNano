"""Cross-validate the designed binder pose against the known VHH
crystal structure in the source PDB.

For each target's best design (by Full-ESMFold2 iptm), this script:
  1. Folds the design with Full ESMFold2
  2. Extracts the predicted CA coordinates for binder and target
  3. Kabsch-aligns the predicted target CA atoms to the PDB target CA atoms
  4. Computes:
       - Target RMSD (sanity: should be near 0)
       - Predicted binder CA vs real VHH CA: backbone binder-RMSD
       - Interface contact RMSD: only contacts within 8 Å of the epitope
  5. Compares to the WT (init) binder pose as a control

Outputs a per-target summary table.

Note on atom extraction
-----------------------
ESMFold2's `out["sample_atom_coords"]` is a flat [n_atoms, 3] tensor.
The atom order is determined by `build_complex_features` and follows
the standard 37-atom per-residue layout (N, CA, C, O, then sidechain).
To get CA, we need to know which atom index within each residue
corresponds to CA. We use `out["atom_to_token"]` (or equivalent) to
map atoms to residues, then take atom index 1 (CA) per residue.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import biotite.structure as bs
import biotite.structure.io.pdb as bs_pdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_target_pdb import setup_target_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


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


def fold_one(model, binder_seq: str, target_seq: str, num_loops: int = 3, num_sampling: int = 14):
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
    return out, features


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray):
    """Kabsch alignment: find R, t that maps P onto Q, return RMSD."""
    assert P.shape == Q.shape
    n, d = P.shape
    cP = P - P.mean(0)
    cQ = Q - Q.mean(0)
    H = cP.T @ cQ
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(0) - P.mean(0) @ R.T
    P_aligned = P @ R.T + t
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(-1).mean()))
    return rmsd, R, t


def decode_atom_name(chars: np.ndarray) -> str:
    """Decode `ref_atom_name_chars` to standard 4-char atom name."""
    return "".join(chr(c + 32) if c > 0 else " " for c in chars).strip()


def get_ca_per_residue(sample_atom_coords: torch.Tensor,
                       atom_to_token: torch.Tensor,
                       ref_atom_name_chars: torch.Tensor,
                       atom_attention_mask: torch.Tensor,
                       binder_len_tokens: int):
    """Extract CA coords per residue, separated by chain.

    Uses `ref_atom_name_chars` to explicitly find the CA atom (avoids relying
    on the "CA is index 1" assumption, which fails for residue 0 of some
    tokens or if N is masked). Returns (binder_ca [n_binder, 3],
    target_ca [n_target, 3]). Skips residues where CA is not found.
    """
    coords = sample_atom_coords[0].cpu().numpy()      # [n_atoms, 3]
    a2t = atom_to_token[0].cpu().numpy()              # [n_atoms]
    chars = ref_atom_name_chars[0].cpu().numpy()      # [n_atoms, 4]
    mask = atom_attention_mask[0].cpu().numpy().astype(bool)

    binder_res_atoms: dict[int, list[int]] = {}
    target_res_atoms: dict[int, list[int]] = {}
    for i in range(len(coords)):
        if not mask[i]:
            continue
        res = int(a2t[i])
        if res < binder_len_tokens:
            binder_res_atoms.setdefault(res, []).append(i)
        else:
            target_res_atoms.setdefault(res, []).append(i)

    binder_ca = []
    for res in sorted(binder_res_atoms.keys()):
        ca_idx = next(
            (i for i in binder_res_atoms[res] if decode_atom_name(chars[i]) == "CA"),
            None,
        )
        if ca_idx is not None:
            binder_ca.append(coords[ca_idx])
    target_ca = []
    for res in sorted(target_res_atoms.keys()):
        ca_idx = next(
            (i for i in target_res_atoms[res] if decode_atom_name(chars[i]) == "CA"),
            None,
        )
        if ca_idx is not None:
            target_ca.append(coords[ca_idx])
    return np.array(binder_ca), np.array(target_ca)


def load_pdb_ca(pdb_path: str, chain: str):
    atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
    ca = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
    return ca.coord.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals-dir", default="/tmp/quick_val/evals")
    p.add_argument("--out", default="/tmp/quick_val/rmsd_summary.json")
    args = p.parse_args()

    target_configs = {
        "PDL1_5JDS_KN035": dict(
            pdb="test/5JDS.pdb", target_chain="A", vhh_chain="B",
            epi=[36,38,43,45,48,50,97,98,99,101,102,103,104,105],
            fw="kn035",
        ),
        "RBD_6WAQ_VHH72": dict(
            pdb="test/6WAQ.pdb", target_chain="B", vhh_chain="A",
            epi=[35,36,37,38,39,40,41,42,43,44,45,46,49,50],
            fw="vhh72",
        ),
        "TNFA_5M2J_ANTITNF": dict(
            pdb="test/5M2J.pdb", target_chain="A", vhh_chain="D",
            epi=[66,67,68,79,80,81,82,83,84,117,118],
            fw="antitnf",
        ),
        "RBD_6ZXN_TY1": dict(
            pdb="/tmp/quick_val/6ZXN_RBD.pdb", target_chain="A", vhh_chain="D",
            epi=[95,96,97,100,101,102,194,195,196,197,198,199,200,201,216,218,219,220,228,229,230,231,232,233,234,235,236,237,238,240,241,242,243,244],
            fw="ty1",
        ),
        "TNFA_5M2M_VHH3": dict(
            pdb="test/5M2M.pdb", target_chain="B", vhh_chain="D",
            epi=[13,14,15,16,17,18,57,58,59,60,61,62,63,64,65,67,94,97,98,100,101,104,105,106,130,131,132,133,134,135,136,137,138,139],
            fw="vhh3",
        ),
    }

    import glob
    model = load_model()
    summary = {}

    for tag, cfg in target_configs.items():
        print(f"\n========= {tag} =========")
        setup = setup_target_design(
            pdb_path=cfg["pdb"], target_chain=cfg["target_chain"],
            epitope_indices=cfg["epi"], framework=cfg["fw"],
        )
        target_seq = setup["target_sequence"]
        binder_wt = setup["binder_full_sequence"]
        binder_len = len(binder_wt)
        cdr = setup["cdr_indices"]
        epi = setup["epitope_token_indices"]
        target_len = len(target_seq)

        # Load real VHH and target CA from PDB
        real_target_ca = load_pdb_ca(cfg["pdb"], cfg["target_chain"])
        real_vhh_ca = load_pdb_ca(cfg["pdb"], cfg["vhh_chain"])
        print(f"  real_target: chain {cfg['target_chain']} ({len(real_target_ca)} CA), "
              f"real_VHH: chain {cfg['vhh_chain']} ({len(real_vhh_ca)} CA)")

        # Find best design across all seeds (by ipTM)
        eval_files = sorted(glob.glob(f"{args.evals_dir}/{tag}_seed*_eval.json"))
        best_iptm = -1.0
        best_design = None
        for ef in eval_files:
            with open(ef) as f:
                d = json.load(f)
            for r in d["results"]:
                if r["name"] == "WT (init)":
                    continue
                if r["iptm"] is not None and r["iptm"] > best_iptm:
                    best_iptm = r["iptm"]
                    best_design = r
                    best_design["source_file"] = ef

        if best_design is None:
            print(f"  [WARN] no non-WT design found in {tag}, skipping RMSD")
            continue
        print(f"  Best design: {best_design['name']} iptm={best_design['iptm']:.3f} "
              f"pTM={best_design['ptm']:.3f} cdr→epi={best_design['cdr_to_epi_min']:.2f}")
        print(f"    cdr_seq: {best_design['cdr_seq']}")
        print(f"    source: {best_design['source_file']}")

        # Fold the best design
        design_seq = best_design["full_seq"]
        wt_seq = binder_wt
        candidates = {
            "WT_init": wt_seq,
            "best_design": design_seq,
        }
        # For comparison, also fold the WT
        per_cand = {}
        for name, seq in candidates.items():
            print(f"\n  Folding {name} (len={len(seq)}) ...")
            try:
                t0 = time.time()
                out, features = fold_one(model, seq, target_seq, 3, 14)
                dt = time.time() - t0
                print(f"    folded in {dt:.1f}s, iptm={float(out['iptm'][0]):.3f}, "
                      f"pTM={float(out['ptm'][0]):.3f}")
            except Exception as e:
                print(f"    [ERR] {e}")
                continue

            # Extract coordinates
            sample_atom_coords = out["sample_atom_coords"]  # [1, n_atoms, 3]
            # The atom_to_token and ref_atom_name_chars are in features dict
            atom_to_token = None
            ref_atom_name_chars = None
            atom_attention_mask = None
            for key in ("atom_to_token", "atom_to_token_map", "atom_idx_to_token_idx"):
                if key in features:
                    atom_to_token = features[key]
                    break
            for key in ("ref_atom_name_chars", "atom_names", "ref_atom_names"):
                if key in features:
                    ref_atom_name_chars = features[key]
                    break
            for key in ("atom_attention_mask", "atom_mask", "ref_atom_mask"):
                if key in features:
                    atom_attention_mask = features[key]
                    break
            if (atom_to_token is None
                    or ref_atom_name_chars is None
                    or atom_attention_mask is None):
                print(f"    [WARN] missing required keys in features. "
                      f"Available: {sorted(features.keys())[:12]}")
                continue

            binder_ca, target_ca = get_ca_per_residue(
                sample_atom_coords, atom_to_token,
                ref_atom_name_chars, atom_attention_mask, binder_len,
            )
            print(f"    pred binder CA: {len(binder_ca)}, target CA: {len(target_ca)}")
            print(f"    real target CA: {len(real_target_ca)}, VHH CA: {len(real_vhh_ca)}")

            # If shapes don't match, truncate to min
            n_target = min(len(target_ca), len(real_target_ca))
            n_vhh = min(len(binder_ca), len(real_vhh_ca))
            target_ca_aligned = target_ca[:n_target]
            real_target_ca_t = real_target_ca[:n_target]
            binder_ca_t = binder_ca[:n_vhh]
            real_vhh_ca_t = real_vhh_ca[:n_vhh]

            # Kabsch-align predicted target to real target
            target_rmsd, R, t = kabsch_rmsd(target_ca_aligned, real_target_ca_t)
            # Apply same transform to binder
            binder_aligned = binder_ca_t @ R.T + t
            binder_rmsd = float(np.sqrt(((binder_aligned - real_vhh_ca_t) ** 2).sum(-1).mean()))

            # Interface contacts: count pairs within 8 Å (binder-target)
            d = np.linalg.norm(binder_aligned[:, None, :] - real_target_ca_t[None, :, :], axis=-1)
            n_interface = int((d < 8.0).sum())
            real_d = np.linalg.norm(real_vhh_ca_t[:, None, :] - real_target_ca_t[None, :, :], axis=-1)
            n_real_interface = int((real_d < 8.0).sum())

            per_cand[name] = {
                "iptm": float(out["iptm"][0]),
                "ptm": float(out["ptm"][0]),
                "binder_rmsd_A": binder_rmsd,
                "target_rmsd_A": target_rmsd,
                "n_pred_interface_contacts": n_interface,
                "n_real_interface_contacts": n_real_interface,
                "fold_time_s": dt,
            }
            print(f"    -> binder_RMSD vs real VHH = {binder_rmsd:.2f} Å")
            print(f"    -> target_RMSD (sanity)  = {target_rmsd:.2f} Å")
            print(f"    -> interface contacts: {n_interface} (pred) vs {n_real_interface} (real)")

        summary[tag] = {
            "config": cfg,
            "best_design": {
                "name": best_design["name"],
                "step": best_design["step"],
                "iptm": best_design["iptm"],
                "ptm": best_design["ptm"],
                "cdr_to_epi_min": best_design["cdr_to_epi_min"],
                "cdr_seq": best_design["cdr_seq"],
                "source_file": best_design["source_file"],
            },
            "pose_comparison": per_cand,
        }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote RMSD summary to {args.out}")

    # Print final table
    print("\n" + "=" * 80)
    print(f"{'target':<22} {'init_iptm':>10} {'design_iptm':>11} {'init_RMSD':>10} "
          f"{'design_RMSD':>11} {'real_RMSD':>10}")
    print("=" * 80)
    for tag, d in summary.items():
        init = d["pose_comparison"].get("WT_init", {})
        design = d["pose_comparison"].get("best_design", {})
        print(f"{tag:<22} {init.get('iptm', 0):>10.3f} {design.get('iptm', 0):>11.3f} "
              f"{init.get('binder_rmsd_A', 0):>10.2f} "
              f"{design.get('binder_rmsd_A', 0):>11.2f}")


if __name__ == "__main__":
    main()
