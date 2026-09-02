"""
Inject PDB coordinates into ESMFold2 confidence head for structure-conditioned scoring.

Key: distogram_atom_idx points to CB (not CA), so we must place ALL backbone atoms,
not just CA. Otherwise the confidence head computes distances from (0,0,0).
"""

from __future__ import annotations

import torch
import numpy as np


def _parse_pdb_all_atoms(pdb_path: str) -> dict[tuple[str, int, str], tuple[float, float, float]]:
    """Parse PDB, return {(chain, res_idx_0based, atom_name): (x, y, z)} for all ATOM lines."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21]
            res_idx = int(line[22:26].strip()) - 1  # 0-based
            atom_name = line[12:16].strip()         # e.g. "CA", "CB", "N", "CG1"
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[(chain, res_idx, atom_name)] = (x, y, z)
    return coords


def build_pdb_coords(
    pdb_path: str,
    features: dict,
    chain_a_len: int,
    chain_b_len: int,
) -> torch.Tensor:
    """Build coordinate tensor [1, 1, n_atoms, 3] from PDB.

    Places ALL atoms found in the PDB at their correct model positions.
    Uses atom name matching: PDB "CA" → model "CA", PDB "CB" → model "CB", etc.

    Chain mapping:
      PDB chain B → nanobody  (feature positions 0 .. chain_a_len-1)
      PDB chain A → antigen    (feature positions chain_a_len .. end)
    """
    pdb_atoms = _parse_pdb_all_atoms(pdb_path)

    n_atoms = features["ref_pos"].shape[1]
    coords = np.zeros((1, 1, n_atoms, 3), dtype=np.float32)

    a2t = features["atom_to_token"][0].cpu().numpy()         # [n_atoms] → residue index
    name_chars = features["ref_atom_name_chars"][0].cpu().numpy()  # [n_atoms, 4]

    # Decode model atom names: chr(code+32) for each char, strip padding
    def _model_atom_name(atom_idx):
        codes = name_chars[atom_idx]
        return "".join(chr(int(c) + 32) if c > 0 else "" for c in codes).strip()

    # PDB chain → (feature_start, feature_end)
    chain_map = [("B", 0, chain_a_len), ("A", chain_a_len, chain_a_len + chain_b_len)]

    for pdb_chain, feat_start, feat_end in chain_map:
        for feat_res in range(feat_start, feat_end):
            pdb_res = feat_res - feat_start  # 0-based in PDB chain

            # Find all model atoms for this residue
            model_atom_indices = np.where(a2t == feat_res)[0]

            for model_idx in model_atom_indices:
                model_name = _model_atom_name(model_idx)
                if not model_name:
                    continue
                pdb_key = (pdb_chain, pdb_res, model_name)
                if pdb_key in pdb_atoms:
                    x, y, z = pdb_atoms[pdb_key]
                    coords[0, 0, model_idx, 0] = x
                    coords[0, 0, model_idx, 1] = y
                    coords[0, 0, model_idx, 2] = z

    return torch.from_numpy(coords)
