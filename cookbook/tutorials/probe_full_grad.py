"""Gradient probe using the actual design code path on Full ESMFold2.

Reuses design_b5_mps_v2.py internals so the probe reflects production conditions.
Tests several Full-model tweaks to recover usable gradients.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

# Import what we need from the existing v2 design
from design_b5_mps_v2 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    MODEL_PATH, ESMC_PATH, DEVICE, NUM_RES_TYPES, TOKENS, AA_TO_TOKEN, AA_DIMS, CYS_TOK,
    AA_FREQ, TEMP_MIN,
)
import binder_design_hy_losses as L
from binder_design_hy_losses import compute_structure_losses
from test_b5_pdb import setup_design


def probe_grad(num_loops=1, disable_msa=True, freeze_trunk_n=0, lr_test=20.0,
               model_path="/Users/huyue/esm-c-fold2/ESMFold2"):
    print(f"\n=== Probe: model={model_path} num_loops={num_loops} "
          f"disable_msa={disable_msa} freeze_trunk_n={freeze_trunk_n} ===", flush=True)

    # Load model (mimic design_b5_mps_v2.load_model)
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    print(f"  loading model from {model_path}...", flush=True)
    t0 = time.time()
    config = ESMFold2Config.from_pretrained(model_path)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(model_path, config=config).float().to(DEVICE).eval()
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)

    for p in model.parameters():
        p.requires_grad_(False)

    # Unwrap @torch.inference_mode()
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"    forward unwrapped for gradient flow", flush=True)

    # Apply tweaks
    if disable_msa:
        # MSAEncoderConfig is a dataclass — replace with new instance
        from transformers.models.esmfold2.configuration_esmfold2 import MSAEncoderConfig
        model.config.msa_encoder = MSAEncoderConfig(enabled=False)
        if hasattr(model, "msa_encoder") and model.msa_encoder is not None:
            model.msa_encoder = None

    if freeze_trunk_n > 0:
        for i, block in enumerate(model.folding_trunk.blocks):
            for p in block.parameters():
                p.requires_grad = (i >= freeze_trunk_n)
        n_train = sum(1 for b in model.folding_trunk.blocks for p in b.parameters() if p.requires_grad)
        n_total = sum(1 for b in model.folding_trunk.blocks for p in b.parameters())
        print(f"  trunk params: {n_train}/{n_total} trainable", flush=True)

    # Get B5 setup
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    binder_wt = setup["binder_full_sequence"]

    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    # Initialize soft logits (WT seed)
    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=3.0).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr_test)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    # Do a single forward + backward to measure gradient
    temperature = 0.5
    res_type_soft = build_soft_res_type(soft_logits, target_one_hot, temperature=temperature)

    cur_seq = soft_to_hard_seq(soft_logits)
    from esmscore._complex import build_complex_features
    feats = build_complex_features(cur_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features["res_type"] = res_type_soft.to(DEVICE)
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}

    print(f"  running fwd+bwd...", flush=True)
    t0 = time.time()
    with torch.set_grad_enabled(True):
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=5,
            num_diffusion_samples=1,
            calculate_confidence=True,
        )
    print(f"    fwd: {time.time()-t0:.1f}s", flush=True)
    disto_bf = out["distogram_logits"].float()
    disto = reorder_bf_to_target_first(disto_bf, binder_len)
    pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

    losses = compute_structure_losses(
        disto, binder_length=binder_len,
        epitope_token_indices=epi, cdr_indices=cdr,
        prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
        n_bins=64, min_dist=2.0, max_dist=22.0,
    )
    lm = aa_freq_loss(soft_logits, mutable_mask)
    total = losses["total_loss"] + 0.01 * lm

    ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
    iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None
    print(f"  losses: total={total.item():.4f} inter={losses['inter_contact_loss'].item():.4f} "
          f"epi={losses['epitope_loss'].item():.4f}", flush=True)
    print(f"  pTM={ptm:.3f} ipTM={iptm:.3f}", flush=True)

    t0 = time.time()
    total.backward()
    print(f"    bwd: {time.time()-t0:.1f}s", flush=True)
    g = soft_logits.grad
    if g is None:
        print("  GRADIENT IS NONE — design loop would not move", flush=True)
        return {"gnorm": 0.0, "gmax": 0.0, "cdr_gnorm": 0.0, "cdr_gmax": 0.0,
                "ptm": ptm, "iptm": iptm}

    gnorm = g.norm().item()
    gmax = g.abs().max().item()
    cdr_t = torch.tensor(cdr, device=DEVICE, dtype=torch.long)
    cdr_g = g[cdr_t]
    cdr_gnorm = cdr_g.norm().item()
    cdr_gmax = cdr_g.abs().max().item()
    print(f"  GRADIENT: full norm={gnorm:.6f} max={gmax:.6f}", flush=True)
    print(f"  CDR norm={cdr_gnorm:.6f} max={cdr_gmax:.6f}", flush=True)

    # Take one optimizer step and see if it changes the sequence
    optimizer.step()
    new_seq = soft_to_hard_seq(soft_logits)
    n_diff = sum(1 for a, b in zip(cur_seq, new_seq) if a != b)
    print(f"  after 1 step (lr={lr_test}): {n_diff} residues changed", flush=True)

    return {"gnorm": gnorm, "gmax": gmax, "cdr_gnorm": cdr_gnorm, "cdr_gmax": cdr_gmax,
            "ptm": ptm, "iptm": iptm, "n_changed": n_diff}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Users/huyue/esm-c-fold2/ESMFold2")
    p.add_argument("--out", default="/tmp/full_grad_probe.json")
    args = p.parse_args()

    results = {}
    configs = [
        ("A_Full_baseline_3loops", dict(num_loops=3, disable_msa=False, freeze_trunk_n=0, lr_test=50.0)),
        ("B_Full_1loop_no_msa",   dict(num_loops=1, disable_msa=True,  freeze_trunk_n=0, lr_test=50.0)),
        ("C_Full_1loop_no_msa_freeze24", dict(num_loops=1, disable_msa=True, freeze_trunk_n=24, lr_test=50.0)),
    ]
    if "Fast" in args.model:
        configs = [
            ("Z_Fast_baseline", dict(num_loops=1, disable_msa=True, freeze_trunk_n=0, lr_test=0.5)),
        ]

    for name, kw in configs:
        try:
            r = probe_grad(model_path=args.model, **kw)
            results[name] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print("\n\n=== Summary ===")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:40s}  ERROR: {r['error']}")
        else:
            print(f"  {name:40s}  cdr_gnorm={r['cdr_gnorm']:.4f}  gmax={r['gmax']:.4f}  "
                  f"ptm={r.get('ptm', 'N/A')} iptm={r.get('iptm', 'N/A')}  "
                  f"n_changed={r.get('n_changed', '?')}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
