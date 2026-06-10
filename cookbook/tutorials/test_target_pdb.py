"""Generalized target setup for v2 epitope-targeted nanobody design.

Mirrors the contract of test_b5_pdb.setup_design, but takes an explicit
target PDB + epitope residue indices instead of a hardcoded B5 path.

Workflow
--------
1. Load the target PDB (chain T = antigen)
2. (Optional) Take a known epitope residue index list (0-based, in chain T).
   If not provided, auto-detect from a "binder chain" B if present in the PDB.
3. Identify CDRs in the binder sequence (via abnumber, Chothia)
4. Build a binder template: framework fixed, CDRs marked "#"
5. Pack CA distances into prior_bins / prior_mask
6. Return the same dict shape as test_b5_pdb.setup_design.

The binder "framework" can be one of:
  - "b5": the B5 VH framework III (127 aa) — default, most tested
  - "vhh72": VHH-72 (127 aa) — from PDB 6WAQ
  - "kn035": KN035 (127 aa) — from PDB 5JDS
  - A full binder sequence string supplied via --binder-sequence

Usage
-----
    # Setup only (no model run)
    python test_target_pdb.py \\
        --target-pdb test/5JDS.pdb --target-chain A \\
        --epitope-indices 28,29,99,100,101,104,108,110,111,112,113 \\
        --framework b5 --mode setup

    # Sanity-test loss functions
    python test_target_pdb.py \\
        --target-pdb test/5JDS.pdb --target-chain A \\
        --epitope-indices 28,29,99,100,101,104,108,110,111,112,113 \\
        --framework b5 --mode sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import biotite.structure as bs
import biotite.structure.io.pdb as bs_pdb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from binder_design_hy_losses import (  # noqa: E402
    MUTABLE_TOKEN,
    _safe_cdr_indices,
    build_pdb_prior,
    compute_epitope_loss,
    compute_structure_losses,
    compute_structure_prior_loss,
    distances_to_bin_indices,
    get_mid_points,
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Canonical VHH / VH framework init sequences. The CDRs are placeholders
# to be replaced by the design loop. Lengths:
#   b5:     127 aa (WT B5 framework III)
#   vhh72:  127 aa (VHH-72 from PDB 6WAQ chain A)
#   kn035:  127 aa (KN035 from PDB 5JDS chain B)
#   antitnf: 115 aa (anti-TNF VHH from PDB 5M2J chain D)
#
# CDR positions (Chothia) for each framework are inferred at runtime by
# abnumber from the placeholder sequence.

INIT_FRAMEWORKS: dict[str, str] = {
    "b5": "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPIYKAPIRWGQGTLVTVS",
    "vhh72": "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSSGS",
    "kn035": "QVQLQESGGGLVQPGGSLRLSCAASGKMSSRRCMAWFRQAPGKERERVAKLLTTSGSTYLADSVKGRFTISQNNAKSTVYLQMNSLKPEDTAMYYCAADSFEDPTCTLVTSSGAFQYWGQGTQVTVS",
    "antitnf": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSNYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLKPEDTALYYCARSPSGFNRGQGTQVTVSS",
    "ty1": "QVQLVETGGGLVQPGGSLRLSCAASGFTFSSVYMNWVRQAPGKGPEWVSRISPNSGNIGYTDSVKGRFTISRDNAKNTLYLQMNNLKPEDTALYYCAIGLNLSSSSVRGQGTQVTVSS",
    "vhh3": "QLQESGGGLVQPGGSLRLSCAASGRTFSDHSGYTYTIGWFRQAPGKEREFVARIYWSSGNTYYADSVKGRFAISRDIAKNTVDLTMNNLEPEDTAVYYCAARDGIPTSRSVESYNYWGQGTQVTVSS",
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
    return [i for i, d in enumerate(min_to_binder) if d <= cutoff]


def parse_epitope(arg: str) -> list[int]:
    """Parse '28,29,99,...' -> [28, 29, 99, ...]."""
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
    epitope_indices: list[int],
    framework: str = "b5",
    binder_sequence: str | None = None,
    epitope_cutoff: float = 8.0,
    prior_min_dist: float = 3.0,
    auto_detect_from_binder: str | None = None,
    binder_chain: str | None = None,
):
    """Build the full design-input dict for an arbitrary target.

    Parameters
    ----------
    pdb_path : str
        Path to the target PDB (may contain only the target, or target+known-VHH).
    target_chain : str
        Chain ID for the target.
    epitope_indices : list[int]
        0-based target residue indices that define the binding epitope.
    framework : str
        Key into INIT_FRAMEWORKS, or a custom sequence if `binder_sequence` is set.
    binder_sequence : str | None
        If given, used as the binder init instead of a framework key.
    auto_detect_from_binder : str | None
        If set (e.g., "B"), auto-detect epitope residues from this binder chain
        in the PDB (residues within `epitope_cutoff` Å of binder heavy atoms).
        Overrides `epitope_indices` if both are given.
    binder_chain : str | None
        If set, read the binder's CA coords from this chain in the PDB
        and use them to build an interface-distance prior (the "v2-B5
        convention"). If None, the prior uses target-target distances
        only (no binder pose constraint).

    Returns
    -------
    dict with the same shape as test_b5_pdb.setup_design:
        target_sequence, binder_template, binder_full_sequence,
        epitope_token_indices, cdr_indices, prior_bins, prior_mask.
    """
    print(f"=== Loading {pdb_path} (target chain {target_chain}) ===")
    target_seq, target_atoms = load_target_chain(pdb_path, target_chain)
    print(f"  Target sequence ({len(target_seq)} aa): {target_seq[:60]}...")

    # --- epitope ---
    if auto_detect_from_binder is not None:
        atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
        binder_ca = atoms[(atoms.chain_id == auto_detect_from_binder)
                          & (atoms.atom_name == "CA")]
        epitope = find_epitope(target_atoms, binder_ca, epitope_cutoff)
        print(f"  Auto-detected epitope (binder chain {auto_detect_from_binder}, "
              f"cutoff {epitope_cutoff} Å): {len(epitope)} residues")
    else:
        epitope = list(epitope_indices)
        print(f"  User-supplied epitope: {len(epitope)} residues: {epitope}")

    if not epitope:
        raise RuntimeError("Epitope is empty. Provide --epitope-indices or "
                           "--auto-detect-binder <chain>.")

    # --- binder init ---
    if binder_sequence is not None:
        binder_seq = binder_sequence
        print(f"  Using custom binder sequence ({len(binder_seq)} aa)")
    else:
        if framework not in INIT_FRAMEWORKS:
            raise ValueError(f"Unknown framework '{framework}'. "
                             f"Choices: {list(INIT_FRAMEWORKS)} or pass "
                             f"--binder-sequence.")
        binder_seq = INIT_FRAMEWORKS[framework]
        print(f"  Using framework '{framework}' init ({len(binder_seq)} aa)")

    # --- CDRs (Chothia) ---
    cdr_indices = _safe_cdr_indices(binder_seq)
    print(f"  CDR positions: {len(cdr_indices)} of {len(binder_seq)}: "
          f"{sorted(cdr_indices)}")
    # Print CDR runs for sanity
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
            print(f"    CDR {s+1}-{e+1}: {binder_seq[s:e+1]}")

    binder_template = make_cdr_mutable(binder_seq, cdr_indices)
    n_mutable = binder_template.count(MUTABLE_TOKEN)
    print(f"  Binder template: {n_mutable} mutable / {len(binder_template)} fixed")

    # --- prior from target + binder CA structure ---
    # target-target distances come from the target chain.
    diff_tt = target_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    target_target_dist = torch.tensor(np.linalg.norm(diff_tt, axis=-1), dtype=torch.float32)

    # interface distances: if we have a binder chain in the PDB, use those
    # real CA coords to build an interface-distance prior. Otherwise leave
    # interface_dist=None and the prior is target-only (no binder pose).
    interface_dist = None
    if binder_chain is not None:
        atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
        binder_ca = atoms[(atoms.chain_id == binder_chain)
                          & (atoms.atom_name == "CA")]
        if len(binder_ca) != len(binder_seq):
            print(f"  [WARN] binder chain {binder_chain} has {len(binder_ca)} CA atoms, "
                  f"but framework '{framework}' has {len(binder_seq)} residues — "
                  f"interface prior may be misaligned")
        diff_bt = binder_ca.coord[:, None, :] - target_atoms.coord[None, :, :]
        interface_dist = torch.tensor(np.linalg.norm(diff_bt, axis=-1), dtype=torch.float32)
        print(f"  Using real PDB interface prior from binder chain {binder_chain} "
              f"({len(binder_ca)} CA atoms)")

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=len(binder_seq),
        target_length=len(target_seq),
        target_target_dist=target_target_dist,
        interface_dist=interface_dist,
        bin_tolerance=prior_min_dist,
    )
    n_constrained = int(prior_mask.sum().item())
    n_iface = int(prior_mask[len(target_seq):, :len(target_seq)].sum().item())
    print(f"  Prior: {n_constrained} pairs constrained "
          f"(target-target={n_constrained - n_iface}, "
          f"interface={n_iface}; L_prior={prior_bins.shape[0]})")

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


def sanity_test(setup):
    """Exercise the loss functions on a synthetic random distogram."""
    print(f"\n=== Sanity test (random distogram) ===")
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    L = len(target_seq) + len(binder_template)
    B = 1
    # 64 bins matches the design loop (ESMFold2-Fast trunk distogram).
    disto_logits = torch.randn(B, L, L, 64, requires_grad=True)

    bin_distance = get_mid_points()  # 64 bins default
    losses = compute_structure_losses(
        disto_logits,
        binder_length=len(binder_template),
        epitope_token_indices=setup["epitope_token_indices"],
        cdr_indices=setup["cdr_indices"],
        prior_bins=setup["prior_bins"],
        prior_mask=setup["prior_mask"],
    )
    print(f"  intra_contact = {losses['intra_contact_loss'].item():.4f}")
    print(f"  inter_contact = {losses['inter_contact_loss'].item():.4f}")
    print(f"  glob          = {losses['glob_loss'].item():.4f}")
    print(f"  epitope       = {losses['epitope_loss'].item():.4f}")
    print(f"  structure_prior= {losses['structure_prior_loss'].item():.4f}")
    print(f"  total         = {losses['total_loss'].item():.4f}")

    losses["total_loss"].backward()
    grad_abs = disto_logits.grad.abs()
    print(f"  grad nonzero positions: {(grad_abs > 0).sum().item()} / {grad_abs.numel()}")
    prior_mask = setup["prior_mask"].unsqueeze(0).unsqueeze(-1)
    grad_on_prior = (grad_abs * prior_mask.float()).sum().item()
    grad_off_prior = (grad_abs * (1 - prior_mask.float())).sum().item()
    print(f"  grad energy on prior region: {grad_on_prior:.2f}, "
          f"off prior: {grad_off_prior:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pdb", required=True, help="Path to target PDB")
    parser.add_argument("--target-chain", required=True, help="Target chain ID")
    parser.add_argument("--epitope-indices", default=None,
                        help="Comma-separated 0-based target residue indices.")
    parser.add_argument("--auto-detect-binder", default=None,
                        help="Chain ID of a binder in the PDB; auto-detect epitope "
                             "from binder heavy atoms within --epitope-cutoff Å.")
    parser.add_argument("--framework", default="b5",
                        choices=list(INIT_FRAMEWORKS),
                        help="Init framework (default: b5)")
    parser.add_argument("--binder-sequence", default=None,
                        help="Override init with a custom sequence")
    parser.add_argument("--epitope-cutoff", type=float, default=8.0)
    parser.add_argument("--prior-min-dist", type=float, default=3.0)
    parser.add_argument("--binder-chain", default=None,
                        help="Chain ID of the binder in the PDB. If set, the prior "
                             "uses real binder CA coords for the interface-distance "
                             "constraint (v2-B5 convention).")
    parser.add_argument("--mode", choices=["setup", "sanity"], default="setup")
    args = parser.parse_args()

    setup = setup_target_design(
        pdb_path=args.target_pdb,
        target_chain=args.target_chain,
        epitope_indices=parse_epitope(args.epitope_indices),
        framework=args.framework,
        binder_sequence=args.binder_sequence,
        epitope_cutoff=args.epitope_cutoff,
        prior_min_dist=args.prior_min_dist,
        auto_detect_from_binder=args.auto_detect_binder,
        binder_chain=args.binder_chain,
    )

    if args.mode == "sanity":
        sanity_test(setup)

    # Print one-line summary for grep
    print(f"\n[OK] setup complete: target_len={len(setup['target_sequence'])} "
          f"binder_len={len(setup['binder_template'])} "
          f"n_cdr={len(setup['cdr_indices'])} "
          f"n_epi={len(setup['epitope_token_indices'])} "
          f"framework={setup['framework']}")


if __name__ == "__main__":
    main()
