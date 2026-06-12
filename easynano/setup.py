"""Target setup: load PDB, identify epitope/CDRs, build prior.

Adapted from ``cookbook/tutorials/test_target_pdb.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import biotite.structure as bs
import biotite.structure.io.pdb as bs_pdb

from .config import (
    INIT_FRAMEWORKS, MUTABLE_TOKEN, EPITOPE_CUTOFF,
    N_BINS, MIN_DIST, MAX_DIST,
)
from .loss import _safe_cdr_indices, build_pdb_prior

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_target_chain(pdb_path: str, target_chain: str):
    """Return (sequence, ca_atoms) for the target chain."""
    atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
    ca = atoms[(atoms.chain_id == target_chain) & (atoms.atom_name == "CA")]
    seq = "".join(THREE_TO_ONE[r] for r in ca.res_name)
    return seq, ca


def find_epitope(target_atoms, binder_atoms, cutoff: float) -> list[int]:
    """0-based target residue indices within `cutoff` Å of any binder heavy atom."""
    diff = target_atoms.coord[None, :, :] - binder_atoms.coord[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    min_to_binder = dist.min(axis=0)
    return [int(i) for i, d in enumerate(min_to_binder) if d <= cutoff]


def parse_epitope(arg: str) -> list[int]:
    if arg is None:
        return []
    return [int(x) for x in arg.split(",") if x.strip()]


def make_cdr_mutable(binder_seq: str, cdr_indices: list[int]) -> str:
    chars = list(binder_seq)
    for i in cdr_indices:
        chars[i] = MUTABLE_TOKEN
    return "".join(chars)


def setup_target_design(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int] | None = None,
    framework: str = "b5",
    binder_sequence: str | None = None,
    epitope_cutoff: float = EPITOPE_CUTOFF,
    prior_min_dist: float = 3.0,
    auto_detect_from_binder: str | None = None,
    binder_chain: str | None = None,
) -> dict:
    """Build the design-input dict for an arbitrary target.

    Returns dict with keys:
        target_sequence, binder_template, binder_full_sequence,
        epitope_token_indices, cdr_indices, prior_bins, prior_mask, framework.
    """
    target_seq, target_atoms = load_target_chain(pdb_path, target_chain)

    # --- epitope ---
    if auto_detect_from_binder is not None:
        atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
        binder_ca = atoms[(atoms.chain_id == auto_detect_from_binder)
                          & (atoms.atom_name == "CA")]
        epitope = find_epitope(target_atoms, binder_ca, epitope_cutoff)
        print(f"  Auto-detected epitope (binder chain {auto_detect_from_binder}, "
              f"cutoff {epitope_cutoff} Å): {len(epitope)} residues")
    else:
        epitope = list(epitope_indices) if epitope_indices else []
        print(f"  User-supplied epitope: {len(epitope)} residues: {epitope[:20]}"
              f"{'...' if len(epitope) > 20 else ''}")

    if not epitope:
        raise RuntimeError("Epitope is empty. Provide epitope_indices or auto_detect_from_binder.")

    # --- binder init ---
    if binder_sequence is not None:
        binder_seq = binder_sequence
        print(f"  Using custom binder sequence ({len(binder_seq)} aa)")
    else:
        if framework not in INIT_FRAMEWORKS:
            raise ValueError(f"Unknown framework '{framework}'. "
                             f"Choices: {list(INIT_FRAMEWORKS)}")
        binder_seq = INIT_FRAMEWORKS[framework]
        print(f"  Using framework '{framework}' init ({len(binder_seq)} aa)")

    # --- CDRs ---
    cdr_indices = _safe_cdr_indices(binder_seq)
    print(f"  CDR positions: {len(cdr_indices)} of {len(binder_seq)}")
    if cdr_indices:
        runs = []
        start = cdr_indices[0]
        prev = start
        for i in cdr_indices[1:]:
            if i != prev + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
        for s, e in runs:
            print(f"    CDR {s + 1}-{e + 1}: {binder_seq[s:e + 1]}")

    binder_template = make_cdr_mutable(binder_seq, cdr_indices)
    n_mutable = binder_template.count(MUTABLE_TOKEN)
    print(f"  Binder template: {n_mutable} mutable / {len(binder_template)} fixed")

    # --- prior ---
    diff_tt = target_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    target_target_dist = torch.tensor(np.linalg.norm(diff_tt, axis=-1), dtype=torch.float32)

    interface_dist = None
    if binder_chain is not None:
        atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
        binder_ca = atoms[(atoms.chain_id == binder_chain)
                          & (atoms.atom_name == "CA")]
        diff_bt = binder_ca.coord[:, None, :] - target_atoms.coord[None, :, :]
        interface_dist = torch.tensor(np.linalg.norm(diff_bt, axis=-1), dtype=torch.float32)
        print(f"  Using real PDB interface prior from binder chain {binder_chain}")

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=len(binder_seq),
        target_length=len(target_seq),
        target_target_dist=target_target_dist,
        interface_dist=interface_dist,
        bin_tolerance=prior_min_dist,
        n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST,
    )
    n_constrained = int(prior_mask.sum().item())
    n_iface = int(prior_mask[len(target_seq):, :len(target_seq)].sum().item())
    print(f"  Prior: {n_constrained} pairs constrained "
          f"(target-target={n_constrained - n_iface}, interface={n_iface})")

    return {
        "target_sequence": target_seq,
        "binder_template": binder_template,
        "binder_full_sequence": binder_seq,
        "epitope_token_indices": epitope,
        "cdr_indices": cdr_indices,
        "prior_bins": prior_bins,
        "prior_mask": prior_mask,
        "framework": framework if binder_sequence is None else "custom",
    }
