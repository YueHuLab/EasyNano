"""Loss functions and PDB-prior helpers for binder design.

Adapted from ``cookbook/tutorials/loss_functions.py``.
Uses ``easynano.config`` for constants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import (
    LOSS_WEIGHTS, EPITOPE_CUTOFF, MUTABLE_TOKEN,
    N_BINS, MIN_DIST, MAX_DIST,
)


# ---- CDR helpers ----

def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs (requires ``abnumber``)."""
    from abnumber import Chain
    chain = Chain(binder_sequence, scheme="chothia")
    indices = []
    for name in ["h1", "h2", "h3"]:
        cdr = getattr(chain, name, None)
        if cdr is None:
            continue
        idx = [pos - 1 for pos in cdr]
        if idx and (not indices or idx[0] > indices[-1]):
            indices.extend(idx)
    return indices


def _safe_cdr_indices(binder_sequence: str) -> list[int]:
    """Return CDR indices, or empty list if abnumber fails."""
    try:
        return _cdr_indices(binder_sequence)
    except Exception:
        return []


# ---- Distance bin helpers ----

def get_mid_points(n_bins: int = N_BINS, min_dist: float = MIN_DIST,
                   max_dist: float = MAX_DIST) -> torch.Tensor:
    """Midpoint of each distance bin in Angstroms."""
    edges = torch.linspace(min_dist, max_dist, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def distances_to_bin_indices(
    distances: torch.Tensor,
    n_bins: int = N_BINS,
    min_dist: float = MIN_DIST,
    max_dist: float = MAX_DIST,
    tolerance: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CA distance matrix to (bin_indices, mask)."""
    bin_width = (max_dist - min_dist) / n_bins
    bin_idx = ((distances - min_dist) / bin_width).long()
    valid = (distances >= min_dist - tolerance) & (distances <= max_dist + tolerance)
    bin_idx = bin_idx.clamp(0, n_bins - 1)
    return bin_idx, valid


# ---- Loss utilities ----

def binned_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of a probability distribution per bin."""
    eps = 1e-8
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def masked_min_k(masked: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Mean of k smallest values per row, ignoring rows with <k valid entries."""
    finite = torch.where(torch.isfinite(masked), masked,
                         torch.full_like(masked, 1e10))
    topk_vals, _ = torch.topk(finite, k, largest=False, dim=-1)
    valid_count = torch.isfinite(masked).sum(dim=-1)
    result = topk_vals.sum(dim=-1) / torch.clamp(valid_count.float(), min=k)
    return result


def masked_average(masked: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(masked)
    return masked.sum(dim=-1) / torch.clamp(finite.sum(dim=-1).float(), min=1)


# ---- Contact losses ----

def compute_contact_loss(
    disto_logits: torch.Tensor,
    contact_mask: torch.Tensor,
    min_sep: int = 0,
    k: int = 2,
    entropy_cutoff: float = 6.5,
    contact_cutoff: float = 14.0,
) -> torch.Tensor:
    """Entropy-based contact loss using distogram logits."""
    midpoints = get_mid_points().to(disto_logits.device)
    probs = F.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    dist_mask = e_dist < contact_cutoff
    L = disto_logits.size(1)
    sep_mask = torch.ones(L, L, dtype=torch.bool, device=disto_logits.device)
    for i in range(L):
        lo = max(0, i - min_sep)
        hi = min(L, i + min_sep + 1)
        sep_mask[i, lo:hi] = False
    entropy = binned_entropy(probs)[0]
    high_entropy = entropy > entropy_cutoff
    mask = contact_mask & dist_mask & sep_mask & high_entropy
    b_entropy_masked = torch.where(mask, entropy, torch.tensor(float("nan")).to(entropy.device))
    row_score = masked_min_k(b_entropy_masked, k=k)
    return row_score.mean()


def compute_intra_contact_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    k: int = 2,
    min_sep: int = 9,
    contact_cutoff: float = 14.0,
) -> torch.Tensor:
    L = disto_logits.size(1)
    contact_mask = torch.zeros(L, L, dtype=torch.bool, device=disto_logits.device)
    start = target_length
    end = target_length + binder_length
    contact_mask[start:end, start:end] = True
    return compute_contact_loss(disto_logits, contact_mask,
                                min_sep=min_sep, k=k, contact_cutoff=contact_cutoff)


def compute_inter_contact_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    k: int = 1,
    min_sep: int = 0,
    contact_cutoff: float = 22.0,
) -> torch.Tensor:
    L = disto_logits.size(1)
    contact_mask = torch.zeros(L, L, dtype=torch.bool, device=disto_logits.device)
    start = target_length
    end = target_length + binder_length
    contact_mask[start:end, :target_length] = True
    contact_mask[:target_length, start:end] = True
    return compute_contact_loss(disto_logits, contact_mask,
                                min_sep=min_sep, k=k, contact_cutoff=contact_cutoff)


def compute_globularity_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
) -> torch.Tensor:
    midpoints = get_mid_points().to(disto_logits.device)
    probs = F.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    start = target_length
    end = target_length + binder_length
    binder_e_dist = e_dist[start:end, start:end]
    mask = torch.eye(binder_length, dtype=torch.bool, device=disto_logits.device)
    pairwise = binder_e_dist[~mask]
    rg_theoretical = 2.86 * (binder_length ** 0.34)
    rg_sq = (pairwise ** 2).mean() / 4.0
    return F.elu(rg_sq - rg_theoretical ** 2) / (rg_theoretical ** 2)


# ---- Epitope loss ----

def compute_epitope_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    epitope_token_indices: list[int],
    cdr_indices: list[int],
    cutoff: float = EPITOPE_CUTOFF,
) -> torch.Tensor:
    """ELU penalty when CDR-to-epitope distances exceed cutoff."""
    midpoints = get_mid_points().to(disto_logits.device)
    probs = F.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    cross = e_dist[target_length:, :target_length]
    cdr_rows = torch.tensor(cdr_indices, device=disto_logits.device).long()
    epi_cols = torch.tensor(epitope_token_indices, device=disto_logits.device).long()
    cdr_to_epi = cross[cdr_rows][:, epi_cols]
    min_per_cdr = cdr_to_epi.min(dim=-1).values
    return F.elu(min_per_cdr - cutoff).mean()


def compute_topk_epitope_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    cdr_indices: list[int],
    k: int = 8,
    cutoff: float = EPITOPE_CUTOFF,
) -> torch.Tensor:
    """Dynamic epitope: picks top-K closest target residues per step."""
    midpoints = get_mid_points().to(disto_logits.device)
    probs = F.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    cross = e_dist[target_length:, :target_length]
    cdr_rows = torch.tensor(cdr_indices, device=disto_logits.device).long()
    cdr_to_all_target = cross[cdr_rows]
    avg_to_each_target = cdr_to_all_target.mean(dim=0)
    _, topk_idx = torch.topk(avg_to_each_target, k=min(k, target_length), largest=False)
    cdr_to_topk = cdr_to_all_target[:, topk_idx]
    min_per_cdr = cdr_to_topk.min(dim=-1).values
    return F.elu(min_per_cdr - cutoff).mean()


# ---- Structure prior loss ----

def compute_structure_prior_loss(
    disto_logits: torch.Tensor,
    prior_bins: torch.Tensor,
    prior_mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(disto_logits, dim=-1)
    target_probs = log_probs.gather(dim=-1, index=prior_bins.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    loss = -target_probs * prior_mask.float()
    return loss.sum() / prior_mask.sum().clamp(min=1)


def build_pdb_prior(
    binder_length: int,
    target_length: int,
    target_target_dist: torch.Tensor,
    interface_dist: torch.Tensor | None = None,
    bin_tolerance: float = 2.5,
    n_bins: int = N_BINS,
    min_dist: float = MIN_DIST,
    max_dist: float = MAX_DIST,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (prior_bins, prior_mask) from CA distance matrices."""
    L = target_length + binder_length
    prior = torch.full((L, L, n_bins), float("nan"))
    for block_dist, mask_i, mask_j in [
        (target_target_dist, slice(0, target_length), slice(0, target_length)),
    ] + ([
        (interface_dist, slice(target_length, target_length + binder_length), slice(0, target_length)),
    ] if interface_dist is not None else []):
        Lb = block_dist.size(0)
        bin_idx, valid = distances_to_bin_indices(
            block_dist, n_bins, min_dist, max_dist, bin_tolerance
        )
        oh = F.one_hot(bin_idx, num_classes=n_bins).float()
        oh[~valid] = float("nan")
        i0 = 0 if mask_i.start is None else mask_i.start
        j0 = 0 if mask_j.start is None else mask_j.start
        prior[i0:i0 + Lb, j0:j0 + Lb, :] = oh
        # symmetric
        if mask_i != mask_j:
            prior[j0:j0 + Lb, i0:i0 + Lb, :] = oh.transpose(0, 1)
    prior_bins = prior.argmax(dim=-1)
    prior_mask = torch.isfinite(prior).any(dim=-1)
    return prior_bins, prior_mask


# ---- Aggregator ----

def compute_structure_losses(
    disto_logits: torch.Tensor,
    binder_length: int,
    epitope_token_indices: list[int],
    cdr_indices: list[int],
    prior_bins: torch.Tensor | None = None,
    prior_mask: torch.Tensor | None = None,
    n_bins: int = N_BINS,
    min_dist: float = MIN_DIST,
    max_dist: float = MAX_DIST,
) -> dict[str, torch.Tensor]:
    target_length = disto_logits.size(1) - binder_length
    losses = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(disto_logits, binder_length, target_length)
    losses["inter_contact_loss"] = compute_inter_contact_loss(disto_logits, binder_length, target_length)
    losses["glob_loss"] = compute_globularity_loss(disto_logits, binder_length, target_length)
    losses["epitope_loss"] = compute_epitope_loss(
        disto_logits, binder_length, target_length,
        epitope_token_indices, cdr_indices,
    )
    if prior_bins is not None and prior_mask is not None:
        losses["structure_prior_loss"] = compute_structure_prior_loss(disto_logits, prior_bins, prior_mask)
    else:
        losses["structure_prior_loss"] = torch.tensor(0.0, device=disto_logits.device)

    total = torch.tensor(0.0, device=disto_logits.device)
    for k, v in losses.items():
        w = LOSS_WEIGHTS.get(k, 0.0)
        total = total + w * v
    losses["total_loss"] = total
    return losses
