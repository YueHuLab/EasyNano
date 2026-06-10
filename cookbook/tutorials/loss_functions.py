"""Loss functions and PDB-prior helpers for binder design.

Pulled out of ``binder_design_hy.py`` so the test scripts can exercise
them locally without pulling in modal / esm / transformers dependencies
that are only present in the cloud image.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- Constants ----

LOSS_WEIGHTS: dict[str, float] = {
    "intra_contact": 0.5,
    "inter_contact": 0.5,
    "glob": 0.2,
    "epitope": 0.05,
    "structure_prior": 0.3,
}
EPITOPE_CUTOFF: float = 8.0  # Angstrom, soft threshold for binder-to-epitope contact

MUTABLE_TOKEN: str = "#"


# ---- Antibody CDR helpers (require abnumber) ----


def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs (requires ``abnumber``)."""
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
    """abnumber wrapper that tolerates MUTABLE_TOKEN (``#``) characters.

    If the sequence contains ``#`` (i.e. a binder template with CDRs
    marked mutable), align the full amino-acid sequence to locate the
    framework, then map the CDR positions back. For fully-defined
    sequences this is a thin wrapper around :func:`_cdr_indices`.
    """
    if MUTABLE_TOKEN not in binder_sequence:
        return _cdr_indices(binder_sequence)
    placeholder = binder_sequence.replace(MUTABLE_TOKEN, "A")
    placeholder_cdrs = set(_cdr_indices(placeholder))
    indices: list[int] = []
    for i, aa in enumerate(binder_sequence):
        if aa == MUTABLE_TOKEN and any(abs(i - j) <= 3 for j in placeholder_cdrs):
            indices.append(i)
    for j in placeholder_cdrs:
        if j < len(binder_sequence) and binder_sequence[j] != MUTABLE_TOKEN:
            indices.append(j)
    return sorted(set(indices))


# ---- Bin midpoint helpers ----


def get_mid_points(n_bins: int = 64, min_dist: float = 2.0, max_dist: float = 22.0) -> torch.Tensor:
    """Distance bin midpoints for the trunk distogram.

    Defaults match ESMFold2-Fast (esm-c-fold2): 64 bins, 2-22 Å. Older
    experimental models used 128 bins / 2-52 Å; pass ``n_bins`` to
    override.
    """
    boundaries = torch.linspace(min_dist, max_dist, n_bins + 1)
    return (boundaries[:-1] + boundaries[1:]) / 2


def distances_to_bin_indices(
    distances: torch.Tensor,
    n_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> torch.Tensor:
    """Convert Å distances to distogram bin indices using nearest-midpoint."""
    midpoints = get_mid_points(n_bins=n_bins, min_dist=min_dist, max_dist=max_dist).to(distances.device)
    return torch.argmin(torch.abs(distances.unsqueeze(-1) - midpoints), dim=-1)


# ---- Core entropy / masking helpers ----


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Entropy of distance distribution within cutoff (design losses only)."""
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in x under mask along the last dimension."""
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Original structure losses (Algorithms 12-13) ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 12: entropy-based contact loss with sequence-separation constraint."""
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
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder-target interface (k=1, min_sep=0, cutoff=22A)."""
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Algorithm 13: radius of gyration vs theoretical packed protein."""
    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


# ---- New: epitope loss (B-scheme) ----


def compute_epitope_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    epitope_token_indices: list[int],
    bin_distance: torch.Tensor,
    cutoff: float = EPITOPE_CUTOFF,
    cdr_indices: list[int] | None = None,
) -> torch.Tensor:
    """Push specified binder residues close to the user-supplied epitope.

    For each binder residue, take the *minimum* expected distance to any
    epitope residue (so each binder residue only needs to find one anchor),
    then apply ELU soft threshold at ``cutoff``.

    - When ``cdr_indices`` is provided (antibody), only CDR residues are
      penalized. Pass ``None`` (or empty) to penalize all binder residues.
    - Returns a scalar loss in Angstroms, averaged over batch and residues.
    """
    if not epitope_token_indices:
        return torch.zeros((), device=distogram_logits.device,
                           dtype=distogram_logits.dtype)

    # binder -> target cross section, then pick epitope columns
    cross = distogram_logits[:, -binder_length:, :-binder_length, :]
    epitope = cross[:, :, epitope_token_indices, :]                 # [B, L_b, E, 128]
    probs = torch.softmax(epitope, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                          # [B, L_b, E]
    min_dist = e_dist.min(dim=-1).values                             # [B, L_b]
    contact_term = F.elu(min_dist - cutoff)                          # [B, L_b]

    if cdr_indices:
        mask = torch.zeros(binder_length, device=distogram_logits.device)
        mask[cdr_indices] = 1.0
        per_res = contact_term * mask[None, :]
        B = distogram_logits.size(0)
        return per_res.sum() / (mask.sum() * B + 1e-8)
    return contact_term.mean()


# ---- New: dynamic (top-K) epitope loss ----


def compute_topk_epitope_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    cdr_indices: list[int],
    bin_distance: torch.Tensor,
    k: int = 8,
    cutoff: float = EPITOPE_CUTOFF,
) -> torch.Tensor:
    """Dynamic-interface epitope loss: at every step, for each CDR residue,
    pick the top-K closest target residues (by distogram expected distance)
    as the "current epitope", then apply the same soft-min-over-K + ELU
    threshold as the fixed-epitope version.

    Unlike ``compute_epitope_loss``, this does NOT take a fixed
    ``epitope_token_indices`` list. The K closest target residues are
    recomputed every step from the predicted distogram.

    Returns a scalar loss in Å, averaged over batch and CDR positions.
    Gradient flows to the (per-residue) K closest target residues only —
    the topk operation is non-smooth in the selected indices, but the
    loss values over those K indices are smooth, so the gradient of the
    min over K is the standard argmin-through-topk pattern.
    """
    full_len = distogram_logits.shape[1]
    target_length = full_len - binder_length
    # binder rows (last binder_length), target cols (first target_length)
    cross = distogram_logits[:, -binder_length:, :-binder_length, :]   # [B, L_b, T, n_bins]
    probs = torch.softmax(cross, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                            # [B, L_b, T]
    # Top-K closest target residues for each binder residue.
    # ``largest=False`` => K smallest expected distances.
    k_eff = min(k, target_length)
    topk_dists, _ = e_dist.topk(k_eff, dim=-1, largest=False)         # [B, L_b, K]
    min_dist = topk_dists.min(dim=-1).values                            # [B, L_b]
    contact_term = F.elu(min_dist - cutoff)                             # [B, L_b]
    # Restrict to CDR positions (in binder-local indexing).
    if cdr_indices:
        mask = torch.zeros(binder_length, device=distogram_logits.device)
        mask[cdr_indices] = 1.0
        per_res = contact_term * mask[None, :]
        B = distogram_logits.size(0)
        return per_res.sum() / (mask.sum() * B + 1e-8)
    return contact_term.mean()


# ---- New: structure-prior loss (PDB anchor) ----


def compute_structure_prior_loss(
    distogram_logits: torch.Tensor,
    prior_bins: torch.Tensor,    # [L, L] int64
    prior_mask: torch.Tensor,    # [L, L] bool
) -> torch.Tensor:
    """Soft structural anchor toward a PDB-derived distogram.

    Per-pair cross-entropy: for each (i, j) where ``prior_mask`` is True,
    this is the negative log-probability the model assigns to the bin
    corresponding to the PDB distance. Minimizing it pulls the predicted
    distogram toward the experimental pose at the constrained region
    (typically target-target and target-binder interface), leaving the
    rest of the complex free to be redesigned.

    Returns scalar nats averaged over batch and constrained pairs.
    """
    if not prior_mask.any():
        return torch.zeros((), device=distogram_logits.device,
                           dtype=distogram_logits.dtype)

    log_probs = F.log_softmax(distogram_logits, dim=-1)
    targets = prior_bins.to(distogram_logits.device).clamp(min=0)
    targets = targets.unsqueeze(0).expand(distogram_logits.size(0), -1, -1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [B, L, L]
    masked = nll * prior_mask.to(distogram_logits.device).unsqueeze(0).float()
    B = distogram_logits.size(0)
    n_pairs = prior_mask.sum().clamp(min=1)
    return masked.sum() / (n_pairs * B)


def build_pdb_prior(
    binder_length: int,
    target_length: int,
    target_target_dist: torch.Tensor | None,    # [L_t, L_t] Å or None
    interface_dist: torch.Tensor | None,         # [L_b, L_t] Å or None (rows=binder, cols=target)
    bin_tolerance: float = 2.0,                  # mask out pairs at this exact distance (numerical noise)
    n_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack CA distance matrices into a (L, L) bin index tensor + bool mask.

    Layout matches the model's distogram convention:
        indices [0, target_length)        -> target residues
        indices [target_length, L)        -> binder residues
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
        b = distances_to_bin_indices(target_target_dist, n_bins=n_bins, min_dist=min_dist, max_dist=max_dist)
        bins[:target_length, :target_length] = b
        m = target_target_dist > bin_tolerance
        m.fill_diagonal_(False)  # self-pairs are always ~0 Å
        mask[:target_length, :target_length] = m

    if interface_dist is not None:
        assert interface_dist.shape == (binder_length, target_length)
        b = distances_to_bin_indices(interface_dist, n_bins=n_bins, min_dist=min_dist, max_dist=max_dist)
        bins[target_length:, :target_length] = b
        bins[:target_length, target_length:] = b.T
        m = interface_dist > bin_tolerance
        mask[target_length:, :target_length] = m
        mask[:target_length, target_length:] = m.T

    return bins, mask


# ---- Aggregate ----


def compute_structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    epitope_token_indices: list[int] | None = None,
    cdr_indices: list[int] | None = None,
    epitope_cutoff: float = EPITOPE_CUTOFF,
    prior_bins: torch.Tensor | None = None,
    prior_mask: torch.Tensor | None = None,
    n_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> dict[str, torch.Tensor]:
    """Compute structural losses and a weighted total.

    - ``epitope_token_indices`` enables a soft "binder must touch these
      target residues" loss. For antibodies, ``cdr_indices`` restricts
      the penalty to the CDRs.
    - ``prior_bins`` / ``prior_mask`` add a soft structural anchor that
      pulls the predicted distogram toward a reference pose (typically
      from an experimental PDB) at the constrained (i, j) pairs.
    - ``n_bins`` / ``min_dist`` / ``max_dist`` match the model's trunk
      distogram. Defaults are ESMFold2-Fast (64 bins, 2-22 Å).
    """
    bin_distance = get_mid_points(n_bins=n_bins, min_dist=min_dist, max_dist=max_dist).to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["inter_contact_loss"] = compute_inter_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["epitope_loss"] = compute_epitope_loss(
        distogram_logits,
        binder_length,
        epitope_token_indices or [],
        bin_distance,
        cutoff=epitope_cutoff,
        cdr_indices=cdr_indices,
    )
    if prior_bins is not None and prior_mask is not None:
        losses["structure_prior_loss"] = compute_structure_prior_loss(
            distogram_logits, prior_bins, prior_mask
        )
    else:
        losses["structure_prior_loss"] = torch.zeros(
            (), device=distogram_logits.device, dtype=distogram_logits.dtype
        )
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    total = total + LOSS_WEIGHTS["epitope"] * losses["epitope_loss"]
    total = total + LOSS_WEIGHTS["structure_prior"] * losses["structure_prior_loss"]
    losses["total_loss"] = total
    return losses
