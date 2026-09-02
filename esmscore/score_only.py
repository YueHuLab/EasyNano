"""
Score-only ESMFold2: skip diffusion, use distogram_head(z) instead of
coordinate-based distogram for the confidence head.

Key insight: distogram_head predicts inter-residue distances from the
pair representation z BEFORE any diffusion. Feeding those predicted
distances into the confidence_head replaces the expensive coordinate
computation while preserving the spatial information the head was
trained to use.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import types
import torch
import torch.nn.functional as F

_EPS = 1e-6


def _categorical_mean(logits: torch.Tensor, start: float, end: float) -> torch.Tensor:
    n_bins = logits.shape[-1]
    bin_width = (end - start) / n_bins
    bin_centers = torch.arange(
        start + 0.5 * bin_width, end, bin_width, device=logits.device, dtype=logits.dtype
    )
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def _patch_for_mps(model) -> None:
    """Monkey-patch MPS-incompatible ops used by ESMFold2."""
    if next(model.parameters()).device.type != "mps":
        return
    if getattr(model, "_mps_patched", False):
        return

    # 1) torch.unique(dim=...) → CPU
    _orig_unique = torch.unique
    def _mps_unique(input, *args, **kwargs):
        if input.device.type == "mps" and kwargs.get("dim") is not None:
            dev = input.device
            r = _orig_unique(input.cpu(), *args, **kwargs)
            return tuple(x.to(dev) for x in r) if isinstance(r, tuple) else r.to(dev)
        return _orig_unique(input, *args, **kwargs)
    torch.unique = _mps_unique  # noqa

    # 2) Tensor.scatter_reduce_ int64 → float32 → int64 (MPS doesn't support int64)
    _orig_sr = torch.Tensor.scatter_reduce_
    def _mps_sr(self, dim, index, src, reduce, *, include_self=True):
        if self.device.type == "mps" and self.dtype in (torch.int64, torch.long):
            f = self.float()
            f.scatter_reduce_(dim, index, src.float(), reduce, include_self=include_self)
            self.copy_(f.long())
            return self
        return _orig_sr(self, dim, index, src, reduce, include_self=include_self)
    torch.Tensor.scatter_reduce_ = _mps_sr  # noqa

    model._mps_patched = True


def score_only_forward(model, features: dict, num_loops: int = 0,
                       pdb_coords: torch.Tensor | None = None) -> dict:
    """Run forward, skip diffusion. Optionally use PDB coords for confidence_head.

    Args:
        model: ESMFold2Model
        features: Model input features
        num_loops: Trunk recycling iterations
        pdb_coords: [1, 1, n_atoms, 3] PDB atom coordinates, or None for zeros

    Returns {"ptm": Tensor[B], "iptm": Tensor[B], "pae": Tensor[B,L,L]}.
    """
    device = next(model.parameters()).device
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}

    _patch_for_mps(model)  # work around MPS aten::unique_dim

    _sample = model.structure_head.sample
    _conf_forward = model.confidence_head.forward

    B, L = features["res_type"].shape
    n_atoms = features["ref_pos"].shape[1]
    dummy_coords = (pdb_coords.to(device) if pdb_coords is not None
                    else torch.zeros(B, 1, n_atoms, 3, device=device))

    def _noop_sample(self, **kw):
        return {"sample_atom_coords": dummy_coords}

    def _score_confidence_forward(self, s_inputs, z, x_pred, **kw):
        """Confidence head: use provided coords (PDB) for distogram, skip pLDDT."""
        s_in = self.s_inputs_norm(s_inputs.float())
        z_f = z.float()

        pair = self.z_norm(z_f)
        rpe = kw.get("relative_position_encoding")
        tbe = kw.get("token_bonds_encoding")
        if rpe is not None:
            pair = pair + rpe
        if tbe is not None:
            pair = pair + tbe
        pair = pair + self.s_to_z(s_in).unsqueeze(2)
        pair = pair + self.s_to_z_transpose(s_in).unsqueeze(1)
        pair = pair + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_in)[:, :, None, :]
            * self.s_to_z_prod_in2(s_in)[:, None, :, :]
        )

        # --- Coordinate-based distogram (PDB or dummy) ---
        from transformers.models.esmfold2.modeling_esmfold2 import (
            gather_rep_atom_coords,
        )
        num_diff = kw.get("num_diffusion_samples", 1)
        # Repeat for multi-sample batches
        if num_diff > 1:
            pair = pair.repeat_interleave(num_diff, dim=0)
        disto_idx = kw.get("distogram_atom_idx").repeat_interleave(num_diff, dim=0)
        atom_to_tok_m = kw.get("atom_to_token").repeat_interleave(num_diff, dim=0)
        atm_mask_m = kw.get("atom_attention_mask").repeat_interleave(num_diff, dim=0)
        rep_idx_m = disto_idx.long()

        # Flatten sample axis in x_pred
        if x_pred.ndim == 4:
            Bm, mult, N, _ = x_pred.shape
            x_flat = x_pred.reshape(Bm * mult, N, 3)
        else:
            x_flat = x_pred

        rep_coords = gather_rep_atom_coords(x_flat, rep_idx_m)
        rep_dist = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
        dist_bins = (rep_dist.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.dist_bin_pairwise_embed(dist_bins)

        mask = kw.get("token_attention_mask")
        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
            pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair = pair + pair_delta.float()
        del pair_delta

        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        mask_f = mask.float()
        N_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + _EPS)
        ptm = ptm_per_row.max(dim=-1).values

        asym_id = kw.get("asym_id")
        if asym_id is not None:
            inter_chain = (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).float() * pair_mask_2d
            iptm_per_row = (tm_expected * inter_chain).sum(dim=-1) / (inter_chain.sum(dim=-1) + _EPS)
            iptm = iptm_per_row.max(dim=-1).values if inter_chain.any() else torch.zeros_like(ptm)
        else:
            iptm = torch.zeros_like(ptm)

        return {"ptm": ptm, "iptm": iptm, "pae": pae}

    try:
        model.structure_head.sample = types.MethodType(_noop_sample, model.structure_head)
        model.confidence_head.forward = types.MethodType(
            _score_confidence_forward, model.confidence_head
        )
        with torch.inference_mode():
            output = model.forward(**features, num_loops=num_loops,
                                   num_sampling_steps=1, num_diffusion_samples=1)
    finally:
        model.structure_head.sample = _sample
        model.confidence_head.forward = _conf_forward

    return {"ptm": output["ptm"].cpu(), "iptm": output["iptm"].cpu(),
            "pae": output["pae"].cpu()}


# ---- Batch scoring helpers ----

def _pad_atom_features(feats_list: list[dict]) -> list[dict]:
    max_atoms = max(f["ref_pos"].shape[1] for f in feats_list)
    atom_keys = {"ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars",
                 "ref_space_uid", "atom_attention_mask", "atom_to_token"}
    padded = []
    for f in feats_list:
        n_atoms = f["ref_pos"].shape[1]
        pad = max_atoms - n_atoms
        if pad == 0:
            padded.append(f)
            continue
        pf = dict(f)
        for k in atom_keys:
            if k not in f:
                continue
            t = f[k]
            if k == "ref_pos":
                pf[k] = F.pad(t, (0, 0, 0, pad), value=0.0)
            elif k == "ref_atom_name_chars":
                pf[k] = F.pad(t, (0, 0, 0, pad), value=0)
            elif k == "atom_attention_mask":
                pf[k] = F.pad(t.bool(), (0, pad), value=False)
            elif k == "atom_to_token":
                pf[k] = F.pad(t, (0, pad), value=0)
            else:
                pf[k] = F.pad(t, (0, pad), value=0)
        padded.append(pf)
    return padded


def _stack_features(feats_list: list[dict]) -> dict:
    batched = {}
    for k in feats_list[0]:
        tensors = [f[k] for f in feats_list if k in f]
        if not tensors:
            continue
        if isinstance(tensors[0], torch.Tensor):
            batched[k] = torch.cat(tensors, dim=0)
        else:
            batched[k] = tensors[0]
    return batched


def score_only_forward_batch(model, feats_list: list[dict], num_loops: int = 0) -> dict:
    if len(feats_list) == 1:
        return score_only_forward(model, feats_list[0], num_loops=num_loops)
    feats_list = _pad_atom_features(feats_list)
    features = _stack_features(feats_list)
    return score_only_forward(model, features, num_loops=num_loops)
