"""Re-evaluate the step-6 candidate from Full-model design_b5_mps_v3_full run.

This sequence was found by Full ESMFold2 (1.3G) design loop at step 6 with
CDR→epi=9.90Å. We re-evaluate at 3-loops/14-sample for an apples-to-apples
comparison with v2's Fast-designed candidates.

Also evaluates the Full-designed sequences to verify the Fast→Full pipeline
outperforms Full-only design.
"""
import json
import time
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from design_b5_mps_v2 import (
    load_model as load_full_model,
    make_target_one_hot,
    reorder_bf_to_target_first,
    align_prior_to_disto, cdr_to_epitope_stats,
)
from binder_design_hy_losses import compute_structure_losses, get_mid_points
from test_b5_pdb import setup_design
import binder_design_hy_losses as L

# This CDR comes from the Full+lr2 design run at step 6.
# Full sequence reconstructed using the binder template with "##########" at CDR positions
# and "######" / "################" for the other CDR runs.
# Original WT: QVQLVESGGGLVQPGGSLRLSCAASGFTFGTGSYYSLGWFRQAPGQGLEAVAAISSSGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARGFTYSYYPDYRAYDFWGQGTLVTVS
# CDR positions (Chothia 1-based): 26-35, 55-60, 102-117
# CDR run mapping (from setup_design output):
#   "##########" at indices 25-34 (CDR-H1, 10 AAs, mapped to positions 26-35 1-based)
#   "######" at indices 54-59 (CDR-H2, 6 AAs, mapped to positions 55-60 1-based)
#   "################" at indices 101-116 (CDR-H3, 16 AAs, mapped to positions 102-117 1-based)

# Step 6 CDR string: EFTFGTGSTFSGSYVSRGFTYSYYPDYCLYDF
# Lengths: 10 + 6 + 16 = 32 ✓
CDR_STEP6 = "EFTFGTGSTFSGSYVSRGFTYSYYPDYCLYDF"

# Reconstruct full sequence by splitting CDR and placing into template
# CDR-H1 (10 AAs): EFTFGTGSTF
# CDR-H2 (6 AAs):  SGSYVS
# CDR-H3 (16 AAs): RGFTYSYYPDYCLYDF
WT_FULL = "QVQLVESGGGLVQPGGSLRLSCAASGFTFGTGSYYSLGWFRQAPGQGLEAVAAISSSGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARGFTYSYYPDYRAYDFWGQGTLVTVS"
from test_b5_pdb import setup_design as _setup_design
TEMPLATE = _setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)["binder_template"]
assert len(TEMPLATE) == 127, f"template len {len(TEMPLATE)} != 127"
# Replace hashes with CDR-H1, then CDR-H2, then CDR-H3 in order
c1, c2, c3 = CDR_STEP6[:10], CDR_STEP6[10:16], CDR_STEP6[16:]
seq = TEMPLATE.replace("##########", c1, 1)
seq = seq.replace("######", c2, 1)
seq = seq.replace("################", c3, 1)
assert len(seq) == len(WT_FULL) == 127, f"length mismatch: {len(seq)}"
print(f"Reconstructed full sequence (len={len(seq)}):")
print(f"  {seq}\n")


def evaluate_sequence(seq, name, model, setup, target_one_hot):
    target_seq = setup["target_sequence"]
    binder_len = len(seq)
    target_len = len(target_seq)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    from esmscore._complex import build_complex_features
    feats = build_complex_features(seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}

    # Build res_type from hard sequence (3D)
    L_t = len(target_seq)
    L_b = len(seq)
    res_type = torch.zeros(1, L_b + L_t, 33, dtype=torch.float32)
    from design_b5_mps_v2 import TOKENS, AA_TO_TOKEN
    for i, aa in enumerate(seq):
        if aa in AA_TO_TOKEN:
            res_type[0, i, AA_TO_TOKEN[aa]] = 1.0
    for i, aa in enumerate(target_seq):
        if aa in AA_TO_TOKEN:
            res_type[0, L_b + i, AA_TO_TOKEN[aa]] = 1.0
    features["res_type"] = res_type

    features = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}

    t0 = time.time()
    with torch.set_grad_enabled(False):
        out = model.forward(
            **features,
            num_loops=3,
            num_sampling_steps=14,
            num_diffusion_samples=1,
            calculate_confidence=True,
        )
    fold_time = time.time() - t0

    disto = out["distogram_logits"].float()
    L_dist = disto.size(1)
    target_len_eff = L_dist - binder_len
    disto_tfb = reorder_bf_to_target_first(disto, binder_len)
    pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto_tfb)
    losses = compute_structure_losses(
        disto_tfb, binder_length=binder_len,
        epitope_token_indices=epi, cdr_indices=cdr,
        prior_bins=pb.to(model.device), prior_mask=pm.to(model.device),
        n_bins=64, min_dist=2.0, max_dist=22.0,
    )
    diag = cdr_to_epitope_stats(disto_tfb, cdr, epi, target_len_eff, binder_len)
    ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
    iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

    return {
        "name": name,
        "cdr_seq": CDR_STEP6,
        "full_seq": seq,
        "ptm": ptm, "iptm": iptm,
        "cdr_to_epi_min": diag["cdr_to_epitope_min"],
        "inter": float(losses["inter_contact_loss"].item()),
        "intra": float(losses["intra_contact_loss"].item()),
        "epi": float(losses["epitope_loss"].item()),
        "prior": float(losses["structure_prior_loss"].item()),
        "total": float(losses["total_loss"].item()),
        "fold_time_s": fold_time,
    }


if __name__ == "__main__":
    print("Loading Full ESMFold2 (1.3G)...\n")
    model = load_full_model()
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    target_one_hot = make_target_one_hot(target_seq, model.device)

    print(f"=== Evaluating step-6 candidate (from Full design) at 3-loops/14-sample ===\n")
    r = evaluate_sequence(seq, "full_step6_EFTFGTGSTFSGSYVSRGFTYSYYPDYCLYDF",
                          model, setup, target_one_hot)
    print(f"  pTM       = {r['ptm']:.4f}")
    print(f"  ipTM      = {r['iptm']:.4f}")
    print(f"  CDR→epi   = {r['cdr_to_epi_min']:.2f} Å")
    print(f"  inter     = {r['inter']:.4f}")
    print(f"  intra     = {r['intra']:.4f}")
    print(f"  epi       = {r['epi']:.4f}")
    print(f"  prior     = {r['prior']:.4f}")
    print(f"  fold_time = {r['fold_time_s']:.1f}s")
    print(f"  CDR       = {r['cdr_seq']}")

    # Save and compare to v2 results
    with open("/tmp/b5_step6_full_eval.json", "w") as f:
        json.dump(r, f, indent=2)

    # Load v2 results for comparison
    with open("/tmp/b5_eval_results.json") as f:
        v2_results = json.load(f)
    print(f"\n=== Comparison with v2 (Fast design + Full eval) ===\n")
    print(f"  {'Name':30s}  {'pTM':>5}  {'ipTM':>5}  {'CDR→epi':>8}  {'inter':>7}  {'epi':>6}")
    for v in v2_results:
        print(f"  {v['name']:30s}  {v['ptm']:>5.3f}  {v['iptm']:>5.3f}  "
              f"{v['cdr_to_epi_min']:>8.2f}  {v['inter']:>7.4f}  {v['epi']:>6.2f}")
    print(f"  {'Full-design step6':30s}  {r['ptm']:>5.3f}  {r['iptm']:>5.3f}  "
          f"{r['cdr_to_epi_min']:>8.2f}  {r['inter']:>7.4f}  {r['epi']:>6.2f}")

    print(f"\nSaved to /tmp/b5_step6_full_eval.json")
