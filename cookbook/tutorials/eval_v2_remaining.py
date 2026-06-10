"""Save final candidate sequences and re-evaluate remaining v2 snapshots.

v2 (Fast design, 100 steps) snapshots are at steps 0,10,20,30,40,50,60,70,80.
Previously evaluated at 3-loops/14-sample: 0 (WT), 50, 60, 70, 80.
Still to evaluate: 10, 20, 30, 40 (cheap, ~1 min each at 3-loops/14-sample).

Also writes a permanent record of all 3 ipTM>0.5 candidates to disk.
"""
import json
import time
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from design_b5_mps_v2 import (
    load_model, make_target_one_hot,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
)
from binder_design_hy_losses import compute_structure_losses
from test_b5_pdb import setup_design


def reconstruct_full(cdr_seq, template):
    """Reconstruct full 127-aa sequence from CDR + template with 'X' holes."""
    c1, c2, c3 = cdr_seq[:10], cdr_seq[10:16], cdr_seq[16:]
    seq = template.replace("#" * 10, c1, 1)
    seq = seq.replace("#" * 6, c2, 1)
    seq = seq.replace("#" * 16, c3, 1)
    return seq


def evaluate_sequence(seq, name, model, setup):
    target_seq = setup["target_sequence"]
    binder_len = len(seq)
    target_len = len(target_seq)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]

    from esmscore._complex import build_complex_features
    from design_b5_mps_v2 import AA_TO_TOKEN
    feats = build_complex_features(seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}

    L_t = len(target_seq)
    res_type = torch.zeros(1, binder_len + L_t, 33, dtype=torch.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_TOKEN:
            res_type[0, i, AA_TO_TOKEN[aa]] = 1.0
    for i, aa in enumerate(target_seq):
        if aa in AA_TO_TOKEN:
            res_type[0, binder_len + i, AA_TO_TOKEN[aa]] = 1.0
    features["res_type"] = res_type
    features = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}

    t0 = time.time()
    with torch.set_grad_enabled(False):
        out = model.forward(
            **features,
            num_loops=3, num_sampling_steps=14, num_diffusion_samples=1,
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
        "name": name, "full_seq": seq, "ptm": ptm, "iptm": iptm,
        "cdr_to_epi_min": diag["cdr_to_epitope_min"],
        "inter": float(losses["inter_contact_loss"].item()),
        "intra": float(losses["intra_contact_loss"].item()),
        "epi": float(losses["epitope_loss"].item()),
        "prior": float(losses["structure_prior_loss"].item()),
        "total": float(losses["total_loss"].item()),
        "fold_time_s": fold_time,
    }


if __name__ == "__main__":
    # 1. Save permanent record of all 3 ipTM>0.5 candidates
    candidates = [
        {
            "name": "step050",
            "rank": 1,
            "cdr_seq": "GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR",
            "full_seq": "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPIYKAPIRWGQGTLVTVS",
            "ptm": 0.799, "iptm": 0.538, "cdr_to_epi_min_A": 10.72,
            "inter": 0.0049, "epi": 2.80,
        },
        {
            "name": "step070",
            "rank": 2,
            "cdr_seq": "GLQIGYGVYMSYSGQSRVVTDSYQPLYKAPIR",
            "full_seq": "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPLYKAPIRWGQGTLVTVS",
            "ptm": 0.789, "iptm": 0.513, "cdr_to_epi_min_A": 11.35,
            "inter": 0.0067, "epi": 3.36,
        },
        {
            "name": "step060",
            "rank": 3,
            "cdr_seq": "GLQIGYGVYMSYSGQSRVVTDSYQPLYKAPIR",
            "full_seq": "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPLYKAPIRWGQGTLVTVS",
            "ptm": 0.787, "iptm": 0.507, "cdr_to_epi_min_A": 11.23,
            "inter": 0.0057, "epi": 3.24,
        },
    ]
    record = {
        "project": "B5 antibody-antigen complex binder design",
        "date": "2026-06-03",
        "design_method": "ESMFold2-Fast (721M) design loop, 100 steps, lr=0.5",
        "eval_method": "ESMFold2 (1.3G) 3-loops/14-sample with confidence head",
        "wt_baseline": {
            "cdr_seq": "GFTFGTGSYYSSSGSSRGFTYSYYPDYRAYDF",
            "ptm": 0.638, "iptm": 0.117, "cdr_to_epi_min_A": 19.08,
        },
        "improvement_vs_wt": {
            "iptm_ratio": 4.6,
            "pTm_ratio": 1.25,
            "cdr_to_epi_delta_A": -8.36,
        },
        "candidates_above_0.5_iptm": candidates,
    }
    perm_path = "/Users/huyue/esmc_design_new/esm-main-2/cookbook/b5_binder_candidates.json"
    with open(perm_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Permanent record saved to {perm_path}")
    print(f"  3 candidates with ipTM > 0.5")
    for c in candidates:
        print(f"    {c['name']}: ipTM={c['iptm']:.3f}  CDR={c['cdr_seq']}")

    # 2. Re-evaluate remaining v2 snapshots
    print(f"\n=== Re-evaluating v2 snapshots at 3-loops/14-sample ===\n")
    model = load_model()
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    template = setup["binder_template"]

    v2_snaps = json.load(open("/tmp/b5_design_v2_snaps.json"))
    existing = json.load(open("/tmp/b5_eval_results.json"))
    existing_names = {e["name"] for e in existing}
    print(f"  already evaluated: {sorted(existing_names)}")

    new_evals = []
    for s in v2_snaps["snapshots"]:
        name = f"step{s['step']:03d}"
        if name in existing_names or s["step"] == 0:
            continue
        full_seq = reconstruct_full(s["cdr_seq"], template)
        r = evaluate_sequence(full_seq, name, model, setup)
        new_evals.append(r)
        print(f"  {name}: pTM={r['ptm']:.3f}  ipTM={r['iptm']:.3f}  "
              f"CDR→epi={r['cdr_to_epi_min']:.2f}  epi={r['epi']:.2f}  "
              f"[{r['fold_time_s']:.1f}s]")

    if new_evals:
        all_results = existing + new_evals
        all_results.sort(key=lambda x: -x["iptm"])
        with open("/tmp/b5_eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  updated /tmp/b5_eval_results.json with {len(new_evals)} new entries")
        print(f"\n  Top 5 by ipTM:")
        for r in all_results[:5]:
            print(f"    {r['name']:30s}  pTM={r['ptm']:.3f}  ipTM={r['iptm']:.3f}  "
                  f"CDR→epi={r['cdr_to_epi_min']:.2f}Å")
    else:
        print("  nothing new to evaluate")
