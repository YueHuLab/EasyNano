"""v9: 3-CDR finetune from v2 step050, with structure_prior built from
the FULL ESMFold2 predicted 3D CA coordinates (NOT distogram expectation).

Differences from v8:
  - v8: prior derived from distogram expected distance E[softmax(d) · midpoints]
  - v9: prior derived from realized 3D CA-CA distances from sample_atom_coords
  - v9 supports num_diffusion_samples > 1 with per-sample averaging (default 4)
    to reduce sampling variance that v8 suffered from (iptm 0.307 → 0.616 noise)

Why v9 might find new optima:
  - Distogram bin width is 0.31 Å (22 Å / 64 bins); CA coords are continuous
  - v8 expected_dist is averaged across the whole distribution → smoother but
    blurs close contacts
  - v9 averages across diffusion samples → variance reduction while keeping
    the actual atomic positions

Pinned: framework 95 positions, mutable: 3 CDRs (32 positions), starting
from v2 step050, lr=0.05, steps=60.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS, MUTABLE_TOKEN,
    compute_structure_losses, get_mid_points, build_pdb_prior,
)
from test_b5_pdb import setup_design  # noqa: E402

from design_b5_mps_v2 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    load_model, MODEL_PATH, ESMC_PATH, DEVICE, NUM_RES_TYPES, TOKENS,
    AA_TO_TOKEN, AA_DIMS, CYS_TOK, AA_FREQ, TEMP_MIN,
    SAMPLE_STEPS_FWD, N_LOOPS_FWD, W_AA_FREQ,
    pin_fixed_positions,
)

# v2 step050 starting point (the classic best)
V2_STEP050_FULL = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAI"
                   "SYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQ"
                   "PIYKAPIRWGQGTLVTVS")

# Atom-name encoding: padded to 4 chars, ord(c) - 32 (space → 0).
# 'C' = 35, 'A' = 33, so CA = [35, 33, 0, 0]
_CA_NAME_CODE = (ord('C') - 32, ord('A') - 32, 0, 0)


def load_full_model():
    print("Loading FULL ESMFold2 (1.3G) for prior prediction ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained("/Users/huyue/esm-c-fold2/ESMFold2")
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(
        "/Users/huyue/esm-c-fold2/ESMFold2", config=config
    ).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def extract_ca_per_token(coords: torch.Tensor,
                          atom_to_token: torch.Tensor,
                          ref_atom_name_chars: torch.Tensor,
                          atom_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return CA coords per token.

    Args:
        coords: [B, n_atoms, 3] predicted atom coordinates (B may include
                diffusion-sample axis if num_diffusion_samples > 1).
        atom_to_token: [n_atoms] int64 token index for each atom.
        ref_atom_name_chars: [n_atoms, 4] atom name as 4 encoded chars.
        atom_mask: [n_atoms] bool, optional validity mask.

    Returns:
        [B, L, 3] CA coords per token. Tokens without a CA atom (none
        for canonical 20 AAs) get NaN; we'll handle them in caller.
    """
    is_ca = (
        (ref_atom_name_chars[:, 0] == _CA_NAME_CODE[0]) &
        (ref_atom_name_chars[:, 1] == _CA_NAME_CODE[1]) &
        (ref_atom_name_chars[:, 2] == _CA_NAME_CODE[2]) &
        (ref_atom_name_chars[:, 3] == _CA_NAME_CODE[3])
    )
    if atom_mask is not None:
        is_ca = is_ca & atom_mask

    ca_atom_idx = is_ca.nonzero(as_tuple=True)[0]
    ca_tok = atom_to_token[ca_atom_idx]
    L = int(atom_to_token.max().item()) + 1
    B = coords.size(0)
    out = torch.full((B, L, 3), float("nan"), device=coords.device, dtype=coords.dtype)
    out[:, ca_tok, :] = coords[:, ca_atom_idx, :]
    return out


def predict_prior_from_full_ca(binder_seq: str, target_seq: str,
                                num_loops: int = 3,
                                num_sampling: int = 14,
                                num_diffusion_samples: int = 4,
                                bin_tolerance: float = 2.5,
                                n_bins: int = 64,
                                min_dist: float = 2.0,
                                max_dist: float = 22.0):
    """Use FULL ESMFold2 to fold (binder, target), extract 3D CA coords
    (averaged across num_diffusion_samples), and build a structure prior.

    Returns: (prior_bins, prior_mask, iptm, ptm)
    """
    print(f"\n=== Predicting prior from CA coords (loops={num_loops}, "
          f"samples={num_sampling}, diffusion={num_diffusion_samples}) ===",
          flush=True)
    print(f"  binder: {binder_seq[:50]}... ({len(binder_seq)} aa)", flush=True)
    print(f"  target: {target_seq[:50]}... ({len(target_seq)} aa)", flush=True)
    model = load_full_model()
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}

    binder_len = len(binder_seq)
    target_len = len(target_seq)
    # atom_to_token / ref_atom_name_chars come from features (B, n_atoms, ...)
    atom_to_token = features["atom_to_token"][0]            # [n_atoms]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]  # [n_atoms, 4]
    atom_mask = features["atom_attention_mask"][0]            # [n_atoms]

    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=num_diffusion_samples,
            calculate_confidence=True,
        )
    print(f"  Full fold took {time.time() - t0:.1f}s", flush=True)
    iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
    print(f"  predicted pTM={ptm:.3f}  ipTM={iptm:.3f}", flush=True)

    sample_coords = out["sample_atom_coords"].float()  # [B*ds, n_atoms, 3]
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]  # experimental path; collapse
    print(f"  sample_coords shape: {tuple(sample_coords.shape)} "
          f"(B*ds × n_atoms × 3)", flush=True)

    ca_coords = extract_ca_per_token(sample_coords, atom_to_token,
                                      ref_atom_name_chars, atom_mask)
    # ca_coords: [B*ds, L, 3] — NaN for missing CA (should be none for canonical AAs)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    print(f"  CA-per-token missing entries: {n_nan}", flush=True)
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_coords = ca_coords.cpu()

    # Average across diffusion samples (and batch) → [L, 3]
    ca_avg = ca_coords.mean(dim=0)
    print(f"  ca_avg shape: {tuple(ca_avg.shape)}", flush=True)

    # Reorder from binder-first (in esmscore._complex it's binder, target) to
    # target-first so prior is consistent with compute_structure_losses.
    perm = torch.cat([torch.arange(target_len, binder_len + target_len),
                       torch.arange(0, target_len)])
    ca_avg = ca_avg[perm]

    # Pairwise CA-CA distances [L, L]
    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]
    L = ca_dist.size(0)
    print(f"  ca_dist shape: {tuple(ca_dist.shape)}, range "
          f"[{ca_dist[ca_dist > 0].min():.2f}, {ca_dist.max():.2f}] Å", flush=True)

    # Disto ordering from build_pdb_prior / compute_structure_losses: [0, target_len) = target,
    # [target_len, L) = binder. ca_avg is now reordered target-first → first
    # target_len rows = target, last binder_len rows = binder. Same as disto.
    tt_dist = ca_dist[:target_len, :target_len]
    iface_dist = ca_dist[target_len:, :target_len]
    print(f"  target-target:  range [{tt_dist[tt_dist > 0].min():.2f}, "
          f"{tt_dist.max():.2f}] Å", flush=True)
    print(f"  interface:      range [{iface_dist[iface_dist > 0].min():.2f}, "
          f"{iface_dist.max():.2f}] Å", flush=True)

    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len,
        target_length=target_len,
        target_target_dist=tt_dist,
        interface_dist=iface_dist,
        bin_tolerance=bin_tolerance,
        n_bins=n_bins, min_dist=min_dist, max_dist=max_dist,
    )
    print(f"  prior_bins shape: {tuple(prior_bins.shape)}", flush=True)
    print(f"  constrained pairs: {int(prior_mask.sum().item())} / "
          f"{prior_mask.numel()}", flush=True)
    print(f"  target-target constrained: "
          f"{int(prior_mask[:target_len, :target_len].sum().item())}", flush=True)
    print(f"  interface constrained:     "
          f"{int(prior_mask[target_len:, :target_len].sum().item())}", flush=True)

    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return prior_bins, prior_mask, iptm, ptm


def run_design(steps: int = 60,
               lr: float = 0.05,
               wt_logit: float = 5.0,
               w_epitope: float = 0.2,
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 4,
               snapshot_every: int = 4,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_v9_cacoord_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V2_STEP050_FULL):
    print(f"=== v9: 3-CDR finetune with CA-coord prior "
          f"(lr={lr}, steps={steps}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]

    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    if prior_bins is None:
        prior_bins = setup["prior_bins"]
    if prior_mask is None:
        prior_mask = setup["prior_mask"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}")
    print(f"  Init: H1={init_seq[25:35]}  H2={init_seq[54:60]}  H3={init_seq[101:117]}")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    template_list = list(init_seq)
    for i in cdr:
        template_list[i] = MUTABLE_TOKEN
    v9_template = "".join(template_list)
    assert v9_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v9_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v9_template, DEVICE)
    mutable_mask = ~fixed_mask
    print(f"  mutable: {int(mutable_mask.sum().item())} (H1=10 + H2=6 + H3=16)  "
          f"fixed: {int(fixed_mask.sum().item())} (framework=95)")

    history = []
    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_inter = float("inf")
    best_seq_inter = ""
    best_step_inter = -1
    best_iptm = -1
    best_seq_iptm = ""
    best_step_iptm = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1

    init_cur = soft_to_hard_seq(soft_logits)
    print(f"\n  init CDR: {''.join(init_cur[i] for i in cdr)}")
    assert init_cur == init_seq, f"init seq != provided init_seq: {init_cur} vs {init_seq}"

    print(f"\nDesigning {steps} steps ...", flush=True)
    header = (f"  {'step':>4}  {'total':>7}  {'epi':>6}  {'inter':>6}  "
              f"{'CDR→epi':>8}  {'pTM':>5}  {'ipTM':>5}  CDR_seq")
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)

        cur_seq = soft_to_hard_seq(soft_logits)
        cdr_set = set(cdr)
        n_cdr_diff = sum(1 for i in cdr_set if cur_seq[i] != init_seq[i])
        non_cdr = [i for i in range(len(cur_seq)) if i not in cdr_set]
        n_fw_diff = sum(1 for i in non_cdr if cur_seq[i] != init_seq[i])
        if n_fw_diff > 0 and step % log_every == 0:
            print(f"  WARNING: framework has {n_fw_diff} diffs at step {step}", flush=True)

        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
                calculate_confidence=True,
            )
        disto_bf = out["distogram_logits"].float()
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        total = losses["total_loss"] + w_aa_freq * lm

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step, "total": float(total.item()),
            "soft_epi": float(losses["epitope_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "n_cdr_diff_from_init": n_cdr_diff,
        }
        history.append(record)

        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>7.3f}  {record['soft_epi']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}  [{elapsed:>5.0f}s]",
                  flush=True)

        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq; best_step_total = step
        if losses["inter_contact_loss"].item() < best_inter:
            best_inter = losses["inter_contact_loss"].item()
            best_seq_inter = cur_seq; best_step_inter = step
        if diag["cdr_to_epitope_min"] < best_cdr_to_epi:
            best_cdr_to_epi = diag["cdr_to_epitope_min"]
            best_seq_epi = cur_seq; best_step_epi = step
        if iptm is not None and iptm > best_iptm:
            best_iptm = iptm; best_seq_iptm = cur_seq; best_step_iptm = step

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            if soft_logits.grad is None:
                print(f"  [WARN step {step}] grad is None, skipping", flush=True)
                continue
            with torch.no_grad():
                soft_logits.grad[fixed_mask] = 0.0
            g_norm = soft_logits.grad.norm().item()
            if step % log_every == 0:
                print(f"  [step {step}] grad_norm={g_norm:.4f}  "
                      f"max={soft_logits.grad.abs().max().item():.4f}  "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, v9_template)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "h1": cdr_seq[:10],
                "h2": cdr_seq[10:16],
                "h3": cdr_seq[16:],
                "full_seq": cur_seq,
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "ptm": ptm, "iptm": iptm,
                "n_cdr_diff_from_init": n_cdr_diff,
            }
            snapshots.append(snap)
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_full_seq": init_seq,
                    "binder_len": binder_len,
                    "config": {
                        "model": "ESMFold2-Fast (721M) for design",
                        "starting_from": "v2 step050",
                        "prior_source": "Full-predicted v2 step050 3D CA coords (averaged over diffusion samples)",
                        "fixed": "framework (95 positions)",
                        "mutable": "H1 (10) + H2 (6) + H3 (16) = 32 positions",
                        "steps": steps, "lr": lr, "wt_logit": wt_logit,
                        "w_epitope": w_epitope, "w_intra": w_intra, "w_inter": w_inter,
                        "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
                        "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
                    },
                    "snapshots": snapshots,
                }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  best total:    step {best_step_total} = {best_total:.3f}")
    print(f"  best CDR→epi:  step {best_step_epi} = {best_cdr_to_epi:.2f}")
    print(f"  best inter:    step {best_step_inter} = {best_inter:.3f}")
    print(f"  best ipTM:     step {best_step_iptm} = {best_iptm:.3f}")
    print(f"  best (by total) CDR: {''.join(best_seq_total[i] for i in cdr)}")
    print(f"  best (by ipTM)  CDR: {''.join(best_seq_iptm[i] for i in cdr)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4,
                   help="num_diffusion_samples for Full prior (averaged over)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v9_cacoord_snaps.json")
    p.add_argument("--init-seq", type=str, default=V2_STEP050_FULL)
    p.add_argument("--use-wt-prior", action="store_true",
                   help="Skip Full prior; use WT crystal prior (sanity check)")
    args = p.parse_args()

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    prior_bins = prior_mask = None
    if not args.use_wt_prior:
        prior_bins, prior_mask, _, _ = predict_prior_from_full_ca(
            args.init_seq, target_seq,
            num_loops=args.full_loops,
            num_sampling=args.full_samples,
            num_diffusion_samples=args.full_diffusion,
        )
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
