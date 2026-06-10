"""Run ESMFold2 (MPS, fp32) on the B5 antigen/nanobody complex and exercise
the new loss functions against the real distogram_logits.

Uses ONLY local weights:
  - /Users/huyue/esm-c-fold2/ESMFold2-Fast   (folding trunk)
  - /Users/huyue/esm-c-fold2/ESMC-6B         (language model, via config.esmc_id)

What this proves (when it runs cleanly):
  1. The local model produces distogram_logits in the format our losses expect.
  2. structure_prior_loss is LOW on the WT pose (model agrees with PDB).
  3. structure_prior_loss is HIGHER on a random/scrambled CDR (sequence
     change perturbs the predicted pose at the prior region).
  4. epitope_loss correctly reports a small value when CDRs sit at the epitope,
     and grows when CDRs are pushed out by mutation.
"""
from __future__ import annotations
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # required by score_only-style code

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Path setup
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                # cookbook/tutorials
sys.path.insert(0, "/Users/huyue/esm-c-fold2")               # esmscore wrapper

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS, EPITOPE_CUTOFF, MUTABLE_TOKEN,
    build_pdb_prior, compute_structure_losses, get_mid_points,
    _safe_cdr_indices,
)
from test_b5_pdb import setup_design  # noqa: E402

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"


def load_model():
    print(f"Loading ESMFold2-Fast from {MODEL_PATH} ...")
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(MODEL_PATH)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(MODEL_PATH, config=config).float().to(DEVICE).eval()
    # MPS patches (scatter_reduce_)
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    print(f"  loaded in {time.time() - t0:.1f}s, params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def fold_complex(model, binder_seq: str, target_seq: str, num_loops: int = 0,
                 num_sampling_steps: int = 1):
    """Forward through ESMFold2; return distogram_logits + a few summary metrics."""
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    feats = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in feats.items() if not k.startswith("_")}
    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(
            **feats,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
        )
    dt = time.time() - t0
    return {
        "distogram_logits": out["distogram_logits"].float().cpu(),  # [B, L, L, 128]
        "ptm": float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None,
        "iptm": float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None,
        "plddt": out["plddt"][0].float().cpu().numpy() if "plddt" in out else None,
        "elapsed_s": dt,
    }


def scramble_cdrs(binder_seq: str, cdr_indices: list[int], rng: np.random.Generator) -> str:
    pool = list("ARNDQEGHILKMFPSTWYV")  # no cys
    chars = list(binder_seq)
    for i in cdr_indices:
        chars[i] = rng.choice(pool)
    return "".join(chars)


def main():
    # Setup (PDB parse + epitope + CDR + prior)
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=3.0)
    target_seq = setup["target_sequence"]
    binder_wt = setup["binder_full_sequence"]   # original B5 nanobody sequence
    target_len = len(target_seq)
    binder_len = len(binder_wt)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    # NOTE: build_complex_features puts BINDER FIRST (chain A = asym 0).
    # Our losses (and build_pdb_prior) expect TARGET first, binder last.
    # Re-pack the prior to put binder at [0, binder_len) and target at [binder_len, L).
    # Easier: rebuild prior in binder-first convention.
    print(f"\n--- Rebuilding prior in binder-first convention (matches build_complex_features) ---")
    # Re-derive distance matrices in the new convention
    import biotite.structure.io.pdb as bs_pdb
    REPO = HERE.parent.parent
    atoms = bs_pdb.PDBFile.read(str(REPO / "test" / "B5.pdb")).get_structure(model=1)
    ca = atoms[atoms.atom_name == "CA"]
    target_atoms = ca[ca.chain_id == "A"]
    binder_atoms = ca[ca.chain_id == "B"]
    diff_tt = target_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    tt_dist = torch.tensor(np.linalg.norm(diff_tt, axis=-1), dtype=torch.float32)
    diff_bb = binder_atoms.coord[:, None, :] - binder_atoms.coord[None, :, :]
    bb_dist = torch.tensor(np.linalg.norm(diff_bb, axis=-1), dtype=torch.float32)
    diff_bt = binder_atoms.coord[:, None, :] - target_atoms.coord[None, :, :]
    bt_dist = torch.tensor(np.linalg.norm(diff_bt, axis=-1), dtype=torch.float32)

    # binder-first layout: [0, binder_len) = binder, [binder_len, L) = target
    # build_pdb_prior signature: (binder_length, target_length, target_target_dist, interface_dist)
    # where interface_dist is [L_b, L_t] (rows=binder, cols=target)
    # and binder is at the END. So we need to manually build for binder-first.
    L = binder_len + target_len
    bins = torch.full((L, L), -1, dtype=torch.long)
    mask = torch.zeros((L, L), dtype=torch.bool)
    midpoints = get_mid_points()
    from binder_design_hy_losses import distances_to_bin_indices
    # target-target -> [binder_len:, binder_len:]
    bins[binder_len:, binder_len:] = distances_to_bin_indices(tt_dist)
    m_tt = tt_dist > 3.0; m_tt.fill_diagonal_(False)
    mask[binder_len:, binder_len:] = m_tt
    # interface (binder-target) -> [0:binder_len, binder_len:]
    bins[:binder_len, binder_len:] = distances_to_bin_indices(bt_dist)
    bins[binder_len:, :binder_len] = bins[:binder_len, binder_len:].T
    m_bt = bt_dist > 3.0
    mask[:binder_len, binder_len:] = m_bt
    mask[binder_len:, :binder_len] = m_bt.T
    prior_bins_bf = bins
    prior_mask_bf = mask

    # Same for our losses: compute_structure_losses uses binder_length and assumes
    # binder at the SUFFIX. We need to call the per-loss functions in a
    # binder-first variant. Easiest: monkey-patch by swapping target/binder in
    # the sequence convention. Since our losses index `distogram_logits[..., -binder_length:, ...]`
    # we instead provide the distogram with binder at the suffix by transposing.

    def reorder_disto_to_target_first(disto_bf: torch.Tensor) -> torch.Tensor:
        """Reorder [B, L, L, 128] from (binder, target) to (target, binder)."""
        perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
        return disto_bf[:, perm, :, :][:, :, perm, :]

    # And re-derive epitope token indices: in target-first they're 0-based in target.
    # That stays the same — we pass target indices to compute_epitope_loss with the
    # TARGET-first reordered distogram.
    epitope_target_indices = epi  # 0-based in target
    cdr_indices = cdr             # 0-based in binder

    print(f"  Target ({target_len} aa), Binder ({binder_len} aa)")
    print(f"  Epitope: {len(epi)} target residues, CDRs: {len(cdr)} binder residues")

    # ---- Load model ----
    model = load_model()

    # ---- Round 1: WT sequence ----
    print(f"\n=== Folding WT nanobody + B5 antigen on MPS ===")
    result_wt = fold_complex(model, binder_wt, target_seq, num_loops=0, num_sampling_steps=1)
    print(f"  forward time: {result_wt['elapsed_s']:.1f}s")
    print(f"  pTM: {result_wt['ptm']:.3f}   ipTM: {result_wt['iptm']:.3f}")
    # Reorder distogram to target-first
    disto_wt = reorder_disto_to_target_first(result_wt["distogram_logits"])
    print(f"  distogram_logits shape (target-first): {tuple(disto_wt.shape)}")

    losses_wt = compute_structure_losses(
        disto_wt,
        binder_length=binder_len,
        epitope_token_indices=epitope_target_indices,
        cdr_indices=cdr_indices,
        prior_bins=prior_bins, prior_mask=prior_mask,  # original target-first prior
    )
    print(f"\n  --- WT losses ---")
    for k, v in losses_wt.items():
        print(f"    {k:>28} = {float(v):.4f}")

    # ---- Round 2: scramble CDRs ----
    print(f"\n=== Folding SCRAMBLED-CDR nanobody + B5 antigen on MPS ===")
    rng = np.random.default_rng(42)
    binder_scrambled = scramble_cdrs(binder_wt, cdr, rng)
    print(f"  WT CDRs (selected): {''.join(binder_wt[i] for i in cdr[:20])}...")
    print(f"  RND CDRs (selected): {''.join(binder_scrambled[i] for i in cdr[:20])}...")
    result_rnd = fold_complex(model, binder_scrambled, target_seq, num_loops=0, num_sampling_steps=1)
    print(f"  forward time: {result_rnd['elapsed_s']:.1f}s")
    print(f"  pTM: {result_rnd['ptm']:.3f}   ipTM: {result_rnd['iptm']:.3f}")
    disto_rnd = reorder_disto_to_target_first(result_rnd["distogram_logits"])

    losses_rnd = compute_structure_losses(
        disto_rnd,
        binder_length=binder_len,
        epitope_token_indices=epitope_target_indices,
        cdr_indices=cdr_indices,
        prior_bins=prior_bins, prior_mask=prior_mask,
    )
    print(f"\n  --- Scrambled-CDR losses ---")
    for k, v in losses_rnd.items():
        print(f"    {k:>28} = {float(v):.4f}")

    # ---- Compare ----
    print(f"\n=== WT vs Scrambled  (Δ = scrambled - WT) ===")
    print(f"  {'metric':>28}    {'WT':>9}  {'RND':>9}    {'Δ':>9}")
    for k in losses_wt:
        wt = float(losses_wt[k]); rnd = float(losses_rnd[k])
        print(f"  {k:>28}    {wt:>9.4f}  {rnd:>9.4f}   {rnd-wt:>+9.4f}")
    print(f"  {'pTM':>28}    {result_wt['ptm']:>9.3f}  "
          f"{result_rnd['ptm']:>9.3f}   {result_rnd['ptm']-result_wt['ptm']:>+9.3f}")
    print(f"  {'ipTM':>28}    {result_wt['iptm']:>9.3f}  "
          f"{result_rnd['iptm']:>9.3f}   {result_rnd['iptm']-result_wt['iptm']:>+9.3f}")
    print(f"\n  LOSS_WEIGHTS: {LOSS_WEIGHTS}")
    print(f"\n  Interpretation:")
    print(f"   - structure_prior_loss should be LOWER for WT (model agrees with PDB pose)")
    print(f"   - epitope_loss should be LOWER for WT (CDRs sit at epitope)")
    print(f"   - intra/inter_contact_loss: scrambled CDRs may lose interface contacts")
    print(f"   - iptm should be HIGHER for WT (model is more confident about WT interface)")


if __name__ == "__main__":
    main()
