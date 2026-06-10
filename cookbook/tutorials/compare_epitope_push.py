"""Compare epitope-push strategies on RBD/VHH72 (6WAQ) — which variant drives cdr_to_epi_min < 8Å?

Variants:
  baseline: w_epi=0.2, cutoff=8.0, w_prior=0.3  (current v9 default)
  A:        w_epi=2.0, cutoff=8.0, w_prior=0.3  (brute-force epitope)
  B:        w_epi=0.2, cutoff=5.0, w_prior=0.3  (extended gradient range)
  C:        w_epi=2.0, cutoff=5.0, w_prior=0.05 (full push, minimal prior)

One seed each, 60 steps. Prior predicted once.
"""
from __future__ import annotations
import os, sys, time, math, json, gc
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from binder_design_hy_losses import (
    LOSS_WEIGHTS, MUTABLE_TOKEN,
    compute_structure_losses, get_mid_points, build_pdb_prior,
)
from test_target_pdb import setup_target_design
from design_target_v9 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    pin_fixed_positions, extract_ca_per_token,
)

# ---- Config ----
MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
FULL_MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33
TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]

AA_FREQ = torch.tensor([
    0.0743, 0.0510, 0.0443, 0.0477, 0.0290, 0.0399, 0.0604, 0.0677,
    0.0227, 0.0554, 0.0968, 0.0580, 0.0221, 0.0394, 0.0444, 0.0580,
    0.0537, 0.0127, 0.0300, 0.0660,
])
_CA_NAME_CODE = (ord('C') - 32, ord('A') - 32, 0, 0)

# RBD/VHH72 config (6WAQ: chain B=RBD ~194aa, chain A=VHH72 ~127aa)
TARGET_PDB = "test/6WAQ.pdb"
TARGET_CHAIN = "B"
EPITOPE_INDICES = [35,36,37,38,39,40,41,42,43,44,45,46,49,50]
FRAMEWORK = "vhh72"
STEPS = 60
SEED = 0

VARIANTS = [
    # (name, w_epi, cutoff, w_prior, wt_logit)
    ("wl1.0_wp0.10", 0.2, 8.0, 0.10, 1.0),
    ("wl1.5_wp0.05", 0.2, 8.0, 0.05, 1.5),
    ("wl2.0_wp0.05", 0.2, 8.0, 0.05, 2.0),
]


def load_model(model_path=MODEL_PATH):
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(model_path)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(model_path, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    return model


def predict_prior(binder_seq, target_seq):
    print(f"\n=== Predicting v9 prior (Full ESMFold2) ===", flush=True)
    model = load_model(FULL_MODEL_PATH)
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    binder_len, target_len = len(binder_seq), len(target_seq)
    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]

    t0 = time.time()
    with torch.inference_mode():
        out = model.forward(**features, num_loops=3, num_sampling_steps=14,
                            num_diffusion_samples=4, calculate_confidence=True)
    print(f"  Full fold: {time.time() - t0:.1f}s  iptm={float(out['iptm'][0].item()):.3f}", flush=True)

    sample_coords = out["sample_atom_coords"].float()
    if sample_coords.dim() == 4:
        sample_coords = sample_coords[:, 0]
    ca_coords = extract_ca_per_token(sample_coords, atom_to_token, ref_atom_name_chars, atom_mask)
    n_nan = int(torch.isnan(ca_coords).any(dim=-1).sum().item())
    if n_nan > 0:
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
    ca_coords = ca_coords.cpu()
    ca_avg = ca_coords.mean(dim=0)

    perm = torch.cat([torch.arange(target_len, binder_len + target_len),
                       torch.arange(0, target_len)])
    ca_avg = ca_avg[perm]
    ca_dist = torch.cdist(ca_avg.unsqueeze(0), ca_avg.unsqueeze(0))[0]
    tt_dist = ca_dist[:target_len, :target_len]
    iface_dist = ca_dist[target_len:, :target_len]
    prior_bins, prior_mask = build_pdb_prior(
        binder_length=binder_len, target_length=target_len,
        target_target_dist=tt_dist, interface_dist=iface_dist,
        bin_tolerance=2.5, n_bins=64, min_dist=2.0, max_dist=22.0,
    )
    del model; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return prior_bins, prior_mask


def run_one_variant(model, setup, prior_bins, prior_mask,
                    w_epi, cutoff, w_prior, seed, wt_logit=5.0):
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    binder_wt = setup["binder_full_sequence"]
    target_len, binder_len = len(target_seq), len(binder_template)
    epi, cdr = setup["epitope_token_indices"], setup["cdr_indices"]
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epi
    LOSS_WEIGHTS["intra_contact"] = 0.5
    LOSS_WEIGHTS["inter_contact"] = 0.5
    LOSS_WEIGHTS["glob"] = 0.2
    LOSS_WEIGHTS["structure_prior"] = w_prior

    torch.manual_seed(seed)
    np.random.seed(seed)
    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit,
                                    pin_cys_in_cdr=True).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=0.05)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    best_cdr_epi = float("inf")
    best_iptm = -1.0
    best_cdr_seq = ""
    final_record = None

    for step in range(STEPS + 1):
        t = (step + 1) / max(STEPS, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = 0.1 + (1 - 0.1) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot, temperature=temperature)
        cur_seq = soft_to_hard_seq(soft_logits)

        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        with torch.set_grad_enabled(True):
            out = model.forward(**features, num_loops=1, num_sampling_steps=5,
                                num_diffusion_samples=1, calculate_confidence=True)
        disto_bf = out["distogram_logits"].float()
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            epitope_cutoff=cutoff, n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        total = losses["total_loss"] + 0.01 * lm

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm_val = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm_val = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)

        if diag["cdr_to_epitope_min"] < best_cdr_epi:
            best_cdr_epi = diag["cdr_to_epitope_min"]
        if iptm_val is not None and iptm_val > best_iptm:
            best_iptm = iptm_val
            best_cdr_seq = cdr_seq
        if step % 10 == 0 or step == STEPS:
            cdr_seq_short = cdr_seq[:16] + "..." if len(cdr_seq) > 19 else cdr_seq
            print(f"    step {step:>3}: cdr→epi={diag['cdr_to_epitope_min']:.1f}Å  "
                  f"iptm={iptm_val:.3f}  CDR={cdr_seq_short}", flush=True)
        if step == STEPS:
            final_record = {
                "cdr_to_epi_min": diag["cdr_to_epitope_min"],
                "iptm": iptm_val, "ptm": ptm_val,
                "cdr_seq": cdr_seq,
                "epi_loss": float(losses["epitope_loss"].item()),
                "inter_loss": float(losses["inter_contact_loss"].item()),
                "intra_loss": float(losses["intra_contact_loss"].item()),
            }

        if step < STEPS:
            optimizer.zero_grad()
            total.backward()
            if soft_logits.grad is None:
                continue
            with torch.no_grad():
                soft_logits.grad[fixed_position_mask(binder_template, DEVICE)] = 0.0
            optimizer.step()
            pin_fixed_positions(soft_logits, binder_template)

    return best_cdr_epi, best_iptm, best_cdr_seq, final_record


def main():
    print("=" * 70)
    print("Epitope-push comparison: Ty1/RBD (6ZXN)")
    print(f"Variants: {[v[0] for v in VARIANTS]}", flush=True)
    print("=" * 70)

    setup = setup_target_design(
        pdb_path=TARGET_PDB, target_chain=TARGET_CHAIN,
        epitope_indices=EPITOPE_INDICES, framework=FRAMEWORK,
    )
    target_seq = setup["target_sequence"]
    binder_wt = setup["binder_full_sequence"]
    print(f"Target: {len(target_seq)}aa  Binder: {len(binder_wt)}aa  "
          f"CDRs: {len(setup['cdr_indices'])}  Epi: {len(EPITOPE_INDICES)}")

    # Predict prior once
    prior_bins, prior_mask = predict_prior(binder_wt, target_seq)

    # Load Fast model once
    model = load_model()

    results = []
    for name, w_epi, cutoff, w_prior, wt_logit in VARIANTS:
        tag = f"{name} (w_epi={w_epi}, cutoff={cutoff}, w_prior={w_prior}, wt_logit={wt_logit})"
        print(f"\n{'='*70}", flush=True)
        print(f"Running: {tag}", flush=True)
        print(f"{'='*70}", flush=True)
        t0 = time.time()
        best_epi, best_iptm, best_cdr, final = run_one_variant(
            model, setup, prior_bins, prior_mask,
            w_epi, cutoff, w_prior, SEED, wt_logit=wt_logit,
        )
        dt = time.time() - t0
        results.append((name, best_epi, best_iptm, best_cdr, final, dt))

        if final:
            print(f"  Done in {dt:.0f}s", flush=True)
            print(f"  Final: cdr→epi={final['cdr_to_epi_min']:.1f}Å  "
                  f"iptm={final['iptm']:.3f}  pTM={final['ptm']:.3f}")
            print(f"  Best:  cdr→epi={best_epi:.1f}Å  iptm={best_iptm:.3f}", flush=True)
            print(f"  CDR:   {final['cdr_seq']}", flush=True)

    # ---- Comparison table ----
    print("\n" + "=" * 70, flush=True)
    print("COMPARISON TABLE", flush=True)
    print("=" * 70, flush=True)
    header = f"  {'Variant':<16} {'cdr→epi':>9} {'iptm':>6} {'pTM':>6} {'epi_loss':>8} {'inter':>7} {'intra':>7} {'CDR':>36}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, best_epi, best_iptm, best_cdr, final, dt in results:
        if final:
            print(f"  {name:<16} {final['cdr_to_epi_min']:>9.1f} {final['iptm']:>6.3f} "
                  f"{final['ptm']:>6.3f} {final['epi_loss']:>8.2f} {final['inter_loss']:>7.3f} "
                  f"{final['intra_loss']:>7.3f} {final['cdr_seq'][:34]}")
        else:
            print(f"  {name:<16} {'FAILED':>9}", flush=True)

    # Key metric: did any variant hit cdr→epi < 8Å?
    print(f"\n{'='*70}", flush=True)
    print("VERDICT")
    print("=" * 70)
    for name, best_epi, best_iptm, best_cdr, final, dt in results:
        if best_epi < 8.0:
            print(f"  {name}: ✓ cdr→epi={best_epi:.1f}Å (TARGET ACHIEVED)", flush=True)
        elif best_epi < 10.0:
            print(f"  {name}: ~ cdr→epi={best_epi:.1f}Å (close)", flush=True)
        else:
            print(f"  {name}: ✗ cdr→epi={best_epi:.1f}Å (still far)", flush=True)


if __name__ == "__main__":
    main()
