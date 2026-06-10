"""End-to-end test: B5.pdb antibody design with epitope + structure prior.

Walks through:
  1. Load test/B5.pdb (chain A = antigen, chain B = antibody VH)
  2. Identify epitope (chain A residues within ``--epitope-cutoff`` Å of chain B)
  3. Identify CDRs in chain B (via abnumber, with MUTABLE_TOKEN tolerance)
  4. Build a custom binder template: framework fixed, CDRs marked ``#``
  5. Pack CA distances into ``prior_bins`` / ``prior_mask`` (build_pdb_prior)
  6. Sanity-test the new loss functions on a synthetic distogram
  7. Print the exact ``modal run`` / local command for the full design

The PDB has:
  - Chain A: 223 residues (antigen, transporter-like, multiple TM helices)
  - Chain B: 127 residues (VH single-domain antibody, framework III)

Usage:
    # 1) Parse + verify setup (no model run, fast)
    python test_b5_pdb.py --mode setup

    # 2) Run the actual design on Modal (cloud H100)
    modal run test_b5_pdb.py

    # 3) Sanity-test only the loss functions on a synthetic distogram
    python test_b5_pdb.py --mode sanity
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

PDB_PATH = str(REPO / "test" / "B5.pdb")
TARGET_CHAIN = "A"
BINDER_CHAIN = "B"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_pdb_chains(pdb_path: str, target_chain: str, binder_chain: str):
    atoms = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
    ca = atoms[atoms.atom_name == "CA"]
    target_atoms = ca[ca.chain_id == target_chain]
    binder_atoms = ca[ca.chain_id == binder_chain]
    target_seq = "".join(THREE_TO_ONE[r] for r in target_atoms.res_name)
    binder_seq = "".join(THREE_TO_ONE[r] for r in binder_atoms.res_name)
    return target_seq, binder_seq, target_atoms, binder_atoms


def find_epitope(target_atoms, binder_atoms, cutoff: float) -> list[int]:
    diff = target_atoms.coord[None, :, :] - binder_atoms.coord[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    min_to_binder = dist.min(axis=0)  # [L_target]
    return [i for i, d in enumerate(min_to_binder) if d <= cutoff]


def make_cdr_mutable(binder_seq: str, cdr_indices: list[int]) -> str:
    chars = list(binder_seq)
    for i in cdr_indices:
        chars[i] = MUTABLE_TOKEN
    return "".join(chars)


def setup_design(epitope_cutoff: float = 8.0, prior_min_dist: float = 3.0):
    """Parse the PDB and assemble all design inputs. Returns a dict."""
    print(f"=== Loading {PDB_PATH} ===")
    target_seq, binder_seq, target_atoms, binder_atoms = load_pdb_chains(
        PDB_PATH, TARGET_CHAIN, BINDER_CHAIN
    )
    print(f"  Target ({TARGET_CHAIN}): {len(target_seq)} aa")
    print(f"  Binder ({BINDER_CHAIN}): {len(binder_seq)} aa  (VH framework III)")
    print(f"    {binder_seq[:50]}...{binder_seq[-20:]}")

    epitope = find_epitope(target_atoms, binder_atoms, epitope_cutoff)
    print(f"\n=== Epitope auto-detect (target within {epitope_cutoff} Å of binder) ===")
    print(f"  {len(epitope)} residues: {epitope}")
    if len(epitope) == 0:
        raise RuntimeError(
            f"No target residue within {epitope_cutoff} Å of binder. "
            "Increase --epitope-cutoff or check the PDB."
        )

    print(f"\n=== CDR identification (abnumber, Chothia) ===")
    cdr_indices = _safe_cdr_indices(binder_seq)
    print(f"  {len(cdr_indices)} CDR positions: {sorted(cdr_indices)}")
    # Sanity: print the actual CDR sequences
    cdr_seqs = []
    runs = []
    if cdr_indices:
        s = cdr_indices[0]
        for i in cdr_indices[1:] + [None]:
            if i is None or i != s + 1:
                runs.append((binder_seq[s:e] if False else binder_seq[s:cdr_indices[cdr_indices.index(s)] + 1 + 1]))
                runs = runs  # placeholder, real runs computed below
                break
            s = i
    # Simpler: just slice the runs
    runs = []
    if cdr_indices:
        start = cdr_indices[0]
        prev = start
        for i in cdr_indices[1:]:
            if i != prev + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
    print(f"  CDR runs (1-based, inclusive):")
    for s, e in runs:
        print(f"    {s+1}-{e+1}: {binder_seq[s:e+1]}")

    binder_template = make_cdr_mutable(binder_seq, cdr_indices)
    n_mutable = binder_template.count(MUTABLE_TOKEN)
    print(f"\n=== Binder template ===")
    print(f"  {binder_template}")
    print(f"  ({n_mutable} mutable positions, {len(binder_template) - n_mutable} fixed)")

    print(f"\n=== Structure prior (CA distance -> distogram bins) ===")
    diff_tt = target_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    target_target_dist = torch.tensor(np.linalg.norm(diff_tt, axis=-1), dtype=torch.float32)
    diff_bt = binder_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    interface_dist = torch.tensor(np.linalg.norm(diff_bt, axis=-1), dtype=torch.float32)
    prior_bins, prior_mask = build_pdb_prior(
        binder_length=len(binder_seq),
        target_length=len(target_seq),
        target_target_dist=target_target_dist,
        interface_dist=interface_dist,
        bin_tolerance=prior_min_dist,
    )
    print(f"  prior_bins shape: {tuple(prior_bins.shape)}  (L={prior_bins.shape[0]})")
    print(f"  constrained pairs: {int(prior_mask.sum().item())} / {prior_mask.numel()}")
    print(f"  target-target pairs constrained: {int(prior_mask[:len(target_seq), :len(target_seq)].sum().item())}")
    print(f"  interface pairs constrained: {int(prior_mask[len(target_seq):, :len(target_seq)].sum().item())}")
    print(f"  interface distance stats (Å):")
    iface_d = interface_dist[interface_dist > prior_min_dist].numpy()
    print(f"    min={iface_d.min():.2f}  median={np.median(iface_d):.2f}  max={iface_d.max():.2f}")
    return {
        "target_sequence": target_seq,
        "binder_template": binder_template,
        "binder_full_sequence": binder_seq,
        "epitope_token_indices": epitope,
        "cdr_indices": cdr_indices,
        "prior_bins": prior_bins,
        "prior_mask": prior_mask,
    }


def sanity_test(setup):
    """Exercise the new losses on a synthetic random distogram."""
    print(f"\n=== Sanity test (random distogram) ===")
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    L = len(target_seq) + len(binder_template)
    B = 1
    disto_logits = torch.randn(B, L, L, 128, requires_grad=True)

    bin_distance = get_mid_points()
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
    # Sanity: grad is non-zero on the prior region
    prior_mask = setup["prior_mask"].unsqueeze(0).unsqueeze(-1)
    grad_on_prior = (grad_abs * prior_mask.float()).sum().item()
    grad_off_prior = (grad_abs * (1 - prior_mask.float())).sum().item()
    print(f"  grad energy on prior region: {grad_on_prior:.2f}, off prior: {grad_off_prior:.2f}")


def show_run_command(setup, mode: str):
    print(f"\n=== How to run the actual design ===")
    print(f"  Mode: {mode}")
    if mode == "modal":
        print(
            "  Submit to Modal (cloud H100). This will run 150 steps, ~10-30 min:"
        )
        print(f"    modal run {Path(__file__).name}")
    elif mode == "local":
        print(
            "  Local H100/GPU. Requires abnumber (conda) + ESMFold2 model weights."
        )
        print(
            f"    uv run {Path(__file__).name} --mode design-local"
        )
    print(f"\n  Inspect the design class in the test:")
    print(f"    python -c \"import sys; sys.path.insert(0, 'cookbook/tutorials');")
    print(f"      from binder_design_hy import ESMFold2Design\"")
    print(f"\n  Once the design completes, the trajectory is logged with")
    print(f"  structure_prior_loss alongside intra/inter/glob/epitope.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["setup", "sanity", "design-modal", "design-local"],
        default="setup",
    )
    parser.add_argument("--epitope-cutoff", type=float, default=8.0,
                        help="Å cutoff for auto-detecting epitope from PDB.")
    parser.add_argument("--prior-min-dist", type=float, default=3.0,
                        help="Mask out PDB pairs closer than this (numerical noise / self).")
    args = parser.parse_args()

    setup = setup_design(
        epitope_cutoff=args.epitope_cutoff,
        prior_min_dist=args.prior_min_dist,
    )

    if args.mode == "setup":
        return
    if args.mode == "sanity":
        sanity_test(setup)
        show_run_command(setup, "modal")
        return
    if args.mode == "design-modal":
        run_modal(setup)
        return
    if args.mode == "design-local":
        run_local(setup)
        return


def run_modal(setup):
    """Submit the design to Modal. Requires ``modal`` CLI and credentials."""
    from binder_design_hy import ESMFold2DesignModal

    app = ESMFold2DesignModal(use_scaling_critics=False)
    seq, trajectory, results = app.design.remote(
        target_sequence=setup["target_sequence"],
        binder_sequence=setup["binder_template"],
        is_antibody=True,
        epitope_token_indices=setup["epitope_token_indices"],
        cdr_indices=setup["cdr_indices"],
        prior_bins=setup["prior_bins"],
        prior_mask=setup["prior_mask"],
        seed=0,
        batch_size=1,
    )
    print(f"\nDesigned sequences: {seq}")
    print(f"Trajectory length: {len(trajectory)} steps")
    for r in results:
        print(f"  final_loss={r['final_loss']:.4f}  "
              f"iptm={r.get('iptm')}  "
              f"distogram_iptm={r.get('distogram_iptm_proxy')}  "
              f"cdr_distogram_iptm={r.get('cdr_distogram_iptm_proxy')}")


def run_local(setup):
    """Run the design locally. Requires GPU + ESMFold2 weights + abnumber (conda)."""
    from binder_design_hy import ESMFold2Design

    app = ESMFold2Design()
    app.load(use_scaling_critics=False)
    seq, trajectory, results = app.design(
        target_sequence=setup["target_sequence"],
        binder_sequence=setup["binder_template"],
        is_antibody=True,
        epitope_token_indices=setup["epitope_token_indices"],
        cdr_indices=setup["cdr_indices"],
        prior_bins=setup["prior_bins"],
        prior_mask=setup["prior_mask"],
        seed=0,
        batch_size=1,
    )
    print(f"\nDesigned sequences: {seq}")
    print(f"Trajectory length: {len(trajectory)} steps")
    for r in results:
        print(f"  final_loss={r['final_loss']:.4f}  "
              f"iptm={r.get('iptm')}  "
              f"distogram_iptm={r.get('distogram_iptm_proxy')}  "
              f"cdr_distogram_iptm={r.get('cdr_distogram_iptm_proxy')}")


if __name__ == "__main__":
    main()
