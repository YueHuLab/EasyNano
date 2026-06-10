"""v8: 3-CDR finetune from v2 step050, but with structure_prior built from
the FULL ESMFold2 predicted complex of v2 step050 + target.

Plan B iteration (per user request):
  v7 confirmed v2 step050 is locally optimal under the WT-crystal prior.
  This v8 uses the FULL model to fold v2 step050 + target once, then
  takes the predicted CA-CA distance matrix (expected distance from
  distogram) as the new structure_prior for the Fast design loop.

Hypothesis: the WT crystal prior (from B5.pdb) is for a different
binder sequence and pulls the design back toward WT contacts.  The
Full-predicted prior for v2 step050 is the binder's "natural" geometry
and should be a more accurate anchor for fine-tuning around v2's basin.

Key differences from v7:
  - Prior: Full-model predicted distogram (not WT crystal)
  - LR: 0.05 (very low — stay in v2's basin)
  - Steps: 60
  - All 3 CDRs free (32 positions), framework pinned (95)
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


def load_full_model():
    """Load the FULL ESMFold2 (1.3G) for prior prediction only."""
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
    # Unwrap @torch.inference_mode() for gradient flow (not needed for inference but
    # keeps a consistent forward signature)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def predict_prior_from_full(binder_seq: str, target_seq: str,
                             num_loops: int = 3, num_sampling: int = 14,
                             bin_tolerance: float = 2.5,
                             n_bins: int = 64, min_dist: float = 2.0, max_dist: float = 22.0):
    """Use FULL ESMFold2 to fold (binder, target) and extract a structure prior.

    Returns: (prior_bins, prior_mask) ready for compute_structure_losses.
    """
    print(f"\n=== Predicting prior with Full ESMFold2 (loops={num_loops}, samples={num_sampling}) ===", flush=True)
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

    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=1,
            calculate_confidence=True,
        )
    print(f"  Full fold took {time.time() - t0:.1f}s", flush=True)
    iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None
    ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
    print(f"  predicted pTM={ptm:.3f}  ipTM={iptm:.3f}", flush=True)

    # disto_bf is in binder-first layout (binder, target). Convert to target-first.
    disto_bf = out["distogram_logits"].float()
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
    disto = disto_bf[:, perm, :, :][:, :, perm, :]

    # Expected distance from distogram
    midpoints = get_mid_points(n_bins=n_bins, min_dist=min_dist, max_dist=max_dist).to(disto.device)
    probs = torch.softmax(disto, dim=-1)
    expected_dist = (probs * midpoints).sum(-1)[0]  # [L, L] in Å, target-first layout
    expected_dist = expected_dist.cpu()
    print(f"  expected_dist shape: {tuple(expected_dist.shape)}, range "
          f"[{expected_dist[expected_dist > 0].min():.2f}, {expected_dist.max():.2f}] Å",
          flush=True)

    # Split into target-target [L_t, L_t] and interface [L_b, L_t] (rows=binder)
    tt_dist = expected_dist[:target_len, :target_len]
    iface_dist = expected_dist[target_len:, :target_len]
    print(f"  target-target:  range [{tt_dist[tt_dist > 0].min():.2f}, {tt_dist.max():.2f}] Å", flush=True)
    print(f"  interface:      range [{iface_dist[iface_dist > 0].min():.2f}, {iface_dist.max():.2f}] Å", flush=True)

    # Pack into bin indices + mask (same format as build_pdb_prior)
    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len,
        target_length=target_len,
        target_target_dist=tt_dist,
        interface_dist=iface_dist,
        bin_tolerance=bin_tolerance,
        n_bins=n_bins, min_dist=min_dist, max_dist=max_dist,
    )
    print(f"  prior_bins shape: {tuple(prior_bins.shape)}", flush=True)
    print(f"  constrained pairs: {int(prior_mask.sum().item())} / {prior_mask.numel()}", flush=True)
    print(f"  target-target constrained: {int(prior_mask[:target_len, :target_len].sum().item())}", flush=True)
    print(f"  interface constrained:     {int(prior_mask[target_len:, :target_len].sum().item())}", flush=True)

    # Free Full model memory
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
               snapshot_path: str = "/tmp/b5_v8_fullprior_snaps.json",
               prior_bins: torch.Tensor | None = None,
               prior_mask: torch.Tensor | None = None,
               init_seq: str = V2_STEP050_FULL):
    print(f"=== v8: 3-CDR finetune with Full-predicted prior (lr={lr}, steps={steps}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]  # has # for all 32 CDR positions

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

    # Build template: WT template with the v2 (or new init) residues at non-CDR positions
    template_list = list(init_seq)
    for i in cdr:
        template_list[i] = MUTABLE_TOKEN
    v8_template = "".join(template_list)
    assert v8_template.count(MUTABLE_TOKEN) == 32, "should be 32 mutable"

    soft_logits = init_soft_logits(v8_template, init_seq, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)

    fixed_mask = fixed_position_mask(v8_template, DEVICE)
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
                print(f"  [step {step}] grad_norm={g_norm:.4f}  max={soft_logits.grad.abs().max().item():.4f}  "
                      f"n_cdr_diff={n_cdr_diff}", flush=True)
            optimizer.step()
            pin_fixed_positions(soft_logits, v8_template)

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
                        "prior_source": "Full-predicted v2 step050 distogram",
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v8_fullprior_snaps.json")
    p.add_argument("--init-seq", type=str, default=V2_STEP050_FULL)
    p.add_argument("--use-wt-prior", action="store_true",
                   help="Skip Full prior; use WT crystal prior (sanity check)")
    args = p.parse_args()

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]

    prior_bins = prior_mask = None
    if not args.use_wt_prior:
        prior_bins, prior_mask, _, _ = predict_prior_from_full(
            args.init_seq, target_seq,
            num_loops=args.full_loops, num_sampling=args.full_samples,
        )
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_prior=args.w_prior,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path,
               prior_bins=prior_bins, prior_mask=prior_mask,
               init_seq=args.init_seq)
