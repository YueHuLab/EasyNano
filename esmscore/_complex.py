"""
Minimal multi-chain feature builder for two-chain protein complexes.

Key differences from naive concatenation:
- residue_index restarts from 0 per chain
- token_index restarts from 0 per chain
- No artificial bond between chains (token_bonds)
- asym_id, entity_id, sym_id distinguish chains
"""

from __future__ import annotations

import torch
from transformers.models.esmfold2.protein_utils import prepare_protein_features


def build_complex_features(chain_a: str, chain_b: str) -> dict:
    """Build properly separated two-chain features.

    chain_a = binder/nanobody (asym=0, entity=1)
    chain_b = target/antigen  (asym=1, entity=2)

    Uses prepare_protein_features for correct atom-level features,
    then fixes all chain-dependent sequence-level features.
    """
    len_a = len(chain_a)
    len_b = len(chain_b)

    # Get features from full concatenated sequence (for correct atom features)
    combined = chain_a + chain_b
    features = prepare_protein_features(combined)

    # --- Fix chain-dependent features ---
    # asym_id: 0 for chain A, 1 for chain B
    asym = torch.zeros_like(features["asym_id"])
    asym[:, len_a:] = 1
    features["asym_id"] = asym

    # entity_id: 1 for chain A, 2 for chain B
    ent = torch.ones_like(features["entity_id"])
    ent[:, len_a:] = 2
    features["entity_id"] = ent

    # sym_id: same as asym for single-copy chains
    features["sym_id"] = asym.clone()

    # residue_index: restart from 0 per chain
    res_idx = torch.zeros_like(features["residue_index"])
    res_idx[:, :len_a] = torch.arange(len_a).unsqueeze(0)
    res_idx[:, len_a:] = torch.arange(len_b).unsqueeze(0)
    features["residue_index"] = res_idx

    # token_index: same as residue_index for protein-only
    features["token_index"] = res_idx.clone()

    # token_bonds: remove artificial cross-chain bond at the junction
    bonds = features["token_bonds"].clone()
    bonds[:, len_a - 1, len_a, :] = 0  # no bond A_last -> B_first
    bonds[:, len_a, len_a - 1, :] = 0  # no bond B_first -> A_last
    features["token_bonds"] = bonds

    features["_len_a"] = len_a
    features["_len_b"] = len_b
    features["_seq_a"] = chain_a
    features["_seq_b"] = chain_b
    return features
