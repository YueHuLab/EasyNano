"""
Convert ESMFold2 full-diffusion output to multi-chain PDB format.

Usage:
    coords, plddt = extract_coords(out, features)
    pdb_str = to_pdb(coords, plddt, features, seq_a, seq_b)
"""

from __future__ import annotations

import torch
from transformers.models.esmfold2.protein_utils import PROTEIN_1TO3

_ELEM = {6: "C", 7: "N", 8: "O", 16: "S"}


def _decode_atom_name(name_chars: torch.Tensor) -> str:
    codes = name_chars.tolist()
    return "".join(chr(c + 32) if c > 0 else "" for c in codes)


def to_pdb(
    coords: torch.Tensor,       # [n_atoms, 3]
    plddt: torch.Tensor,        # [L] per-residue
    atom_to_token: torch.Tensor,  # [n_atoms]
    asym_id: torch.Tensor,      # [L]
    residue_index: torch.Tensor, # [L]
    ref_atom_name_chars: torch.Tensor,  # [n_atoms, 4]
    ref_element: torch.Tensor,  # [n_atoms]
    atom_mask: torch.Tensor,    # [n_atoms] bool
    seq_a: str,                 # chain A sequence (asym=0)
    seq_b: str,                 # chain B sequence (asym=1)
) -> str:
    """Write multi-chain PDB string with pLDDT as B-factor.

    Chain A (nanobody) → chain ID 'A', residues 1..len(seq_a)
    Chain B (antigen)  → chain ID 'B', residues 1..len(seq_b)
    """
    chain_map = {0: ("A", seq_a, 0), 1: ("B", seq_b, len(seq_a))}

    # Per-chain residue counters (1-based)
    chain_res_idx = {"A": 0, "B": 0}
    last_token = -1

    atoms = []
    for atom_i in range(len(coords)):
        if not atom_mask[atom_i]:
            continue
        token = int(atom_to_token[atom_i].item())
        if token >= len(asym_id):
            continue

        chain_val = int(asym_id[token].item())
        if chain_val not in chain_map:
            continue
        chain_id, seq, seq_offset = chain_map[chain_val]

        # Residue info
        if token != last_token:
            chain_res_idx[chain_id] += 1
            last_token = token
        res_num = chain_res_idx[chain_id]
        res_name_3 = PROTEIN_1TO3.get(seq[token - seq_offset], "UNK")

        # Atom info
        name = _decode_atom_name(ref_atom_name_chars[atom_i])
        elem = _ELEM.get(int(ref_element[atom_i].item()), "C")
        x, y, z = coords[atom_i, 0].item(), coords[atom_i, 1].item(), coords[atom_i, 2].item()
        b_factor = float(plddt[token].item())

        serial = len(atoms) + 1
        # PDB format: ATOM/HETATM aligned columns
        line = (
            f"ATOM  {serial:5d} {name:^4s} {res_name_3:3s} {chain_id:1s}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}          {elem:>2s}"
        )
        atoms.append(line)

    # TER between chains
    chain_a_end = None
    for i, line in enumerate(atoms):
        if chain_a_end is None and line[21] == "B":
            chain_a_end = i
            break
    if chain_a_end:
        ter_line = f"TER   {chain_a_end + 1:5d}      {atoms[chain_a_end][17:20]} {atoms[chain_a_end][21]:1s}{atoms[chain_a_end][22:26]}"
        atoms.insert(chain_a_end, ter_line)
        # renumber after TER
        for j in range(chain_a_end + 1, len(atoms)):
            old = atoms[j]
            atoms[j] = f"ATOM  {j + 1:5d}" + old[11:]

    # END
    atoms.append(f"TER   {len(atoms) + 1:5d}      {atoms[-1][17:20]} {atoms[-1][21]:1s}{atoms[-1][22:26]}")
    atoms.append("END")

    return "\n".join(atoms) + "\n"


def predict_structure(
    model,
    features: dict,
    num_loops: int = 1,
    num_sampling_steps: int = 64,
) -> dict:
    """Run full diffusion, return coords + all confidence metrics.

    Returns dict with keys:
        coords, plddt, pae, ptm, iptm, atom_to_token, asym_id,
        residue_index, ref_atom_name_chars, ref_element, atom_mask,
        pdb_str, seq_a, seq_b
    """
    device = next(model.parameters()).device
    fd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
          for k, v in features.items() if not k.startswith("_")}

    with torch.inference_mode():
        out = model.forward(**fd, num_loops=num_loops,
                            num_sampling_steps=num_sampling_steps,
                            num_diffusion_samples=1)

    coords = out["sample_atom_coords"][0].cpu()
    plddt = out["plddt"][0].cpu()
    pae = out["pae"][0].cpu()
    atom_mask = out["atom_pad_mask"][0].cpu().bool()

    result = {
        "coords": coords,
        "plddt": plddt,
        "pae": pae,
        "ptm": float(out["ptm"][0].item()),
        "iptm": float(out["iptm"][0].item()),
        "atom_to_token": fd["atom_to_token"][0].cpu(),
        "asym_id": fd["asym_id"][0].cpu(),
        "residue_index": fd["residue_index"][0].cpu(),
        "ref_atom_name_chars": fd["ref_atom_name_chars"][0].cpu(),
        "ref_element": fd["ref_element"][0].cpu(),
        "atom_mask": atom_mask,
        "seq_a": features["_seq_a"],
        "seq_b": features["_seq_b"],
    }

    result["pdb_str"] = to_pdb(
        result["coords"], result["plddt"],
        result["atom_to_token"], result["asym_id"],
        result["residue_index"], result["ref_atom_name_chars"],
        result["ref_element"], result["atom_mask"],
        result["seq_a"], result["seq_b"],
    )
    return result
