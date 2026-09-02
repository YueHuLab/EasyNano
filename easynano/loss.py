"""Loss functions and PDB-prior helpers for binder design.

Adapted from ``cookbook/tutorials/loss_functions.py`` (the implementation the
manuscript experiments used). Uses ``easynano.config`` for constants.

Key fixes vs. the previous easynano version:
  - ``compute_structure_losses`` aggregator now uses explicit key mapping
    (the old ``LOSS_WEIGHTS.get(k, 0.0)`` returned 0 for every loss because
    the dict keys lacked the ``_loss`` suffix, so ``total_loss`` was always 0).
  - ``_safe_cdr_indices`` now tolerates ``MUTABLE_TOKEN`` (``#``) characters
    in binder templates (required during gradient-based design).
  - ``build_pdb_prior`` now masks the diagonal self-pairs (distance ~0 Å
    would otherwise produce a bogus bin-0 prior for every residue).
  - ``compute_structure_prior_loss`` now expands the prior to the full batch
    instead of only gathering from batch index 0.
  - All structure losses preserve the batch dimension (no ``[0]`` indexing),
    matching the cookbook implementation.
  - Globularity constant aligned to ``2.38 * n**0.365`` (cookbook).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import (
    EPITOPE_CUTOFF,
    LOSS_WEIGHTS,
    MAX_DIST,
    MIN_DIST,
    MUTABLE_TOKEN,
    N_BINS,
)

# ---- Antibody CDR helpers (require abnumber) ----

def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs (requires ``abnumber``).

    Mirrors ``cookbook/tutorials/loss_functions.py::_cdr_indices``: uses
    ``_anarci_align`` so that both heavy and light chains are recovered when
    present, and CDR positions are located by substring search.
    """
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain("".join(result[i][0].values()), scheme="chothia")
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def _safe_cdr_indices(binder_sequence: str) -> list[int]:
    """Return CDR indices, tolerating ``MUTABLE_TOKEN`` (``#``) characters.

    If the sequence contains ``#`` (a binder template with CDRs marked
    mutable), align the placeholder sequence (``#`` -> ``A``) to locate the
    framework, then map CDR positions back and include any ``#`` that falls
    within +/-3 residues of a placeholder CDR position. For fully-defined
    sequences this is a thin wrapper around :func:`_cdr_indices`.
    """
    if MUTABLE_TOKEN not in binder_sequence:
        try:
            return _cdr_indices(binder_sequence)
        except Exception:
            return []
    placeholder = binder_sequence.replace(MUTABLE_TOKEN, "A")
    try:
        placeholder_cdrs = set(_cdr_indices(placeholder))
    except Exception:
        return []
    indices: list[int] = []
    for i, aa in enumerate(binder_sequence):
        if aa == MUTABLE_TOKEN and any(abs(i - j) <= 3 for j in placeholder_cdrs):
            indices.append(i)
    for j in placeholder_cdrs:
        if j < len(binder_sequence) and binder_sequence[j] != MUTABLE_TOKEN:
            indices.append(j)
    return sorted(set(indices))


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
    """Convert CA distance matrix to (bin_indices, valid_mask).

    Uses nearest-midpoint binning (same as cookbook) so that a distance lands
    in the bin whose midpoint is closest. ``valid`` marks pairs whose distance
    falls within ``[min_dist - tolerance, max_dist + tolerance]``.
    """
    midpoints = get_mid_points(n_bins=n_bins, min_dist=min_dist,
                               max_dist=max_dist).to(distances.device)
    bin_idx = torch.argmin(torch.abs(distances.unsqueeze(-1) - midpoints), dim=-1)
    valid = (distances >= min_dist - tolerance) & (distances <= max_dist + tolerance)
    return bin_idx, valid


# ---- Core entropy / masking helpers (batch-preserving) ----

def binned_entropy(
    dgram: torch.Tensor,
    bin_distance: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Entropy of the distance distribution within ``cutoff`` per pair.

    ``dgram`` is the distogram logits ``[B, L, L, n_bins]``;
    ``bin_distance`` is the per-bin midpoint distance ``[n_bins]``.
    Returns ``[B, L, L]``.
    """
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in ``x`` under ``mask`` along the last dim."""
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along the last axis."""
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Contact losses ----

def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Entropy-based contact loss with sequence-separation constraint.

    ``chain_mask`` / ``binder_mask`` are ``[L]`` float tensors. Returns a
    scalar averaged over the batch.
    """
    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    k: int = 2,
    min_sep: int = 9,
    contact_cutoff: float = 14.0,
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""
    bin_distance = get_mid_points().to(disto_logits.device)
    full_len = disto_logits.size(1)
    is_binder = torch.zeros(full_len, device=disto_logits.device)
    start = target_length
    is_binder[start:start + binder_length] = 1.0
    return compute_contact_loss(
        disto_logits, bin_distance,
        num_contacts=k, min_sep=min_sep, cutoff=contact_cutoff,
        chain_mask=is_binder, binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    k: int = 1,
    min_sep: int = 0,
    contact_cutoff: float = 22.0,
) -> torch.Tensor:
    """Binder-target interface (k=1, min_sep=0, cutoff=22A)."""
    bin_distance = get_mid_points().to(disto_logits.device)
    full_len = disto_logits.size(1)
    is_binder = torch.zeros(full_len, device=disto_logits.device)
    start = target_length
    is_binder[start:start + binder_length] = 1.0
    return compute_contact_loss(
        disto_logits, bin_distance,
        num_contacts=k, min_sep=min_sep, cutoff=contact_cutoff,
        chain_mask=1 - is_binder, binder_mask=is_binder,
    )


def compute_globularity_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
) -> torch.Tensor:
    """Radius of gyration vs. theoretical packed protein.

    Uses ``Rg_th = 2.38 * n**0.365`` (cookbook). Binder block is the last
    ``binder_length`` rows/cols (target-first layout).
    """
    bin_distance = get_mid_points().to(disto_logits.device)
    binder_disto = disto_logits[:, target_length:, target_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n ** 0.365)
    return F.elu(rg_term - rg_th)


# ---- Epitope loss ----

def compute_epitope_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    epitope_token_indices: list[int],
    cdr_indices: list[int],
    cutoff: float = EPITOPE_CUTOFF,
) -> torch.Tensor:
    """ELU penalty when CDR-to-epitope distances exceed cutoff.

    For each binder residue, take the minimum expected distance to any
    epitope residue, then apply an ELU soft threshold at ``cutoff``. When
    ``cdr_indices`` is provided, only CDR residues are penalized.
    Returns a scalar averaged over batch and residues.
    """
    bin_distance = get_mid_points().to(disto_logits.device)
    if not epitope_token_indices:
        return torch.zeros((), device=disto_logits.device,
                           dtype=disto_logits.dtype)

    # binder rows (last binder_length), target cols (first target_length)
    cross = disto_logits[:, target_length:, :target_length, :]
    epitope = cross[:, :, epitope_token_indices, :]            # [B, L_b, E, n_bins]
    probs = torch.softmax(epitope, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                     # [B, L_b, E]
    min_dist = e_dist.min(dim=-1).values                        # [B, L_b]
    contact_term = F.elu(min_dist - cutoff)                     # [B, L_b]

    if cdr_indices:
        mask = torch.zeros(binder_length, device=disto_logits.device)
        mask[cdr_indices] = 1.0
        per_res = contact_term * mask[None, :]
        B = disto_logits.size(0)
        return per_res.sum() / (mask.sum() * B + 1e-8)
    return contact_term.mean()


def compute_topk_epitope_loss(
    disto_logits: torch.Tensor,
    binder_length: int,
    target_length: int,
    cdr_indices: list[int],
    k: int = 8,
    cutoff: float = EPITOPE_CUTOFF,
) -> torch.Tensor:
    """Dynamic epitope: picks top-K closest target residues per step."""
    bin_distance = get_mid_points().to(disto_logits.device)
    cross = disto_logits[:, target_length:, :target_length, :]   # [B, L_b, T, n_bins]
    probs = torch.softmax(cross, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                      # [B, L_b, T]
    k_eff = min(k, target_length)
    topk_dists, _ = e_dist.topk(k_eff, dim=-1, largest=False)    # [B, L_b, K]
    min_dist = topk_dists.min(dim=-1).values                     # [B, L_b]
    contact_term = F.elu(min_dist - cutoff)                      # [B, L_b]
    if cdr_indices:
        mask = torch.zeros(binder_length, device=disto_logits.device)
        mask[cdr_indices] = 1.0
        per_res = contact_term * mask[None, :]
        B = disto_logits.size(0)
        return per_res.sum() / (mask.sum() * B + 1e-8)
    return contact_term.mean()


# ---- Structure prior loss ----

def compute_structure_prior_loss(
    disto_logits: torch.Tensor,
    prior_bins: torch.Tensor,
    prior_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft structural anchor toward a PDB-derived distogram.

    Per-pair cross-entropy: for each (i, j) where ``prior_mask`` is True,
    this is the negative log-probability the model assigns to the bin
    corresponding to the PDB distance. Returns scalar nats averaged over
    batch and constrained pairs.
    """
    if not prior_mask.any():
        return torch.zeros((), device=disto_logits.device,
                           dtype=disto_logits.dtype)

    log_probs = F.log_softmax(disto_logits, dim=-1)
    targets = prior_bins.to(disto_logits.device).clamp(min=0)
    targets = targets.unsqueeze(0).expand(disto_logits.size(0), -1, -1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [B, L, L]
    masked = nll * prior_mask.to(disto_logits.device).unsqueeze(0).float()
    B = disto_logits.size(0)
    n_pairs = prior_mask.sum().clamp(min=1)
    return masked.sum() / (n_pairs * B)


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
    """Build (prior_bins, prior_mask) from CA distance matrices.

    Layout: indices ``[0, target_length)`` -> target residues,
    ``[target_length, L)`` -> binder residues (target-first layout).

    Self-pairs on the diagonal (distance ~0 Å) are always masked out so they
    do not produce a bogus bin-0 prior.
    """
    L = target_length + binder_length
    device = (
        target_target_dist.device
        if target_target_dist is not None
        else (interface_dist.device if interface_dist is not None else torch.device("cpu"))
    )
    bins = torch.full((L, L), fill_value=-1, dtype=torch.long, device=device)
    mask = torch.zeros((L, L), dtype=torch.bool, device=device)

    if target_target_dist is not None:
        assert target_target_dist.shape == (target_length, target_length)
        b = distances_to_bin_indices(
            target_target_dist, n_bins=n_bins, min_dist=min_dist, max_dist=max_dist
        )[0]
        bins[:target_length, :target_length] = b
        m = target_target_dist > bin_tolerance
        m.fill_diagonal_(False)  # self-pairs are always ~0 Å
        mask[:target_length, :target_length] = m

    if interface_dist is not None:
        assert interface_dist.shape == (binder_length, target_length)
        b = distances_to_bin_indices(
            interface_dist, n_bins=n_bins, min_dist=min_dist, max_dist=max_dist
        )[0]
        bins[target_length:, :target_length] = b
        bins[:target_length, target_length:] = b.T
        m = interface_dist > bin_tolerance
        mask[target_length:, :target_length] = m
        mask[:target_length, target_length:] = m.T

    return bins, mask


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
    """Compute structural losses and a weighted total.

    The aggregator uses explicit key mapping (not ``LOSS_WEIGHTS.get(k, 0)``)
    so that the ``*_loss`` keys in the returned dict are correctly weighted by
    the ``LOSS_WEIGHTS`` entries (which use keys without the ``_loss`` suffix).
    ``LOSS_WEIGHTS`` may be overridden at runtime by ``design.py``.
    """
    target_length = disto_logits.size(1) - binder_length
    losses: dict[str, torch.Tensor] = {}
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
        losses["structure_prior_loss"] = torch.zeros(
            (), device=disto_logits.device, dtype=disto_logits.dtype
        )

    B = disto_logits.size(0)
    total = torch.zeros(B, device=disto_logits.device, dtype=disto_logits.dtype)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    total = total + LOSS_WEIGHTS["epitope"] * losses["epitope_loss"]
    total = total + LOSS_WEIGHTS["structure_prior"] * losses["structure_prior_loss"]
    losses["total_loss"] = total
    return losses
