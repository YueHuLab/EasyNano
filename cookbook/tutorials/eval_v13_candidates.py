"""Evaluate v13 H3-only candidates with Full ESMFold2."""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    compute_structure_losses, get_mid_points,
)
from test_b5_pdb import setup_design  # noqa: E402

DEVICE = "mps"


def load_model_full():
    print("Loading FULL ESMFold2 (1.3G) for evaluation ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained("/Users/huyue/esm-c-fold2/ESMFold2")
    config.esmc_id = "/Users/huyue/esm-c-fold2/ESMC-6B"
    model = ESMFold2Model.from_pretrained(
        "/Users/huyue/esm-c-fold2/ESMFold2", config=config
    ).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def fold_one(model, binder_seq, target_seq, num_loops=3, num_sampling=14):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    with torch.inference_mode():
        out = model.forward(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling,
            num_diffusion_samples=1,
            calculate_confidence=True,
        )
    return out


def load_snaps(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["snapshots"], data["init_full_seq"]


def evaluate(snaps, out_path, init_seq, include_init=True,
             label_prefix="v13 step", init_label="v9 step 48 (init)"):
    print(f"=== Evaluating {len(snaps)} {label_prefix} candidates with Full model ===\n",
          flush=True)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_len = len(init_seq)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    target_len = len(target_seq)

    model = load_model_full()
    results = []

    eval_list = []
    if include_init:
        eval_list.append((init_label, init_seq))
    for snap in snaps:
        eval_list.append((f"{label_prefix} {snap['step']}", snap["full_seq"]))

    print(f"  {'Name':>22}  {'pTM':>6}  {'ipTM':>6}  {'CDR→epi':>8}  "
          f"{'inter':>7}  {'epi':>7}  CDR", flush=True)
    print(f"  {'-'*120}", flush=True)

    for name, full in eval_list:
        cdr_seq = "".join(full[i] for i in cdr)
        try:
            t0 = time.time()
            out = fold_one(model, full, target_seq, num_loops=3, num_sampling=14)
            dt = time.time() - t0
        except Exception as e:
            print(f"  [ERR] {name}: {e}", flush=True)
            continue
        disto_bf = out["distogram_logits"].float()
        L = disto_bf.size(1)
        perm = torch.cat([torch.arange(binder_len, L), torch.arange(0, binder_len)])
        disto = disto_bf[:, perm, :, :][:, :, perm, :]
        L_p = prior_bins.size(0)
        pb, pm = prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]
        if disto.size(1) != L_p:
            disto = disto[:, :L_p, :L_p, :]
        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb, prior_mask=pm,
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        midpoints = get_mid_points().to(disto.device)
        probs = torch.softmax(disto, dim=-1)
        e_dist = (probs * midpoints).sum(-1)[0]
        cross = e_dist[target_len:, :target_len]
        cdr_to_e = cross[cdr][:, epi]
        cdr_min = cdr_to_e.min(dim=-1).values.mean().item()
        ptm = float(out["ptm"][0].item()) if out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if out["iptm"].numel() else None
        record = {
            "name": name, "cdr_seq": cdr_seq, "full_seq": full,
            "ptm": ptm, "iptm": iptm,
            "cdr_to_epi_min": cdr_min,
            "inter": float(losses["inter_contact_loss"].item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "epi": float(losses["epitope_loss"].item()),
            "prior": float(losses["structure_prior_loss"].item()),
            "total": float(losses["total_loss"].item()),
            "fold_time_s": dt,
        }
        results.append(record)
        ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
        iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
        print(f"  {name:>22}  {ptm_s:>6}  {iptm_s:>6}  {cdr_min:>8.2f}  "
              f"{record['inter']:>7.3f}  {record['epi']:>7.3f}  {cdr_seq}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {len(results)} results to {out_path}", flush=True)

    print(f"\n=== Top by ipTM ===", flush=True)
    for r in sorted(results, key=lambda r: -(r["iptm"] or 0))[:5]:
        print(f"  {r['name']}: ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
              f"CDR→epi={r['cdr_to_epi_min']:.2f}Å  "
              f"H1={r['cdr_seq'][:10]}/H2={r['cdr_seq'][10:16]}/H3={r['cdr_seq'][16:]}",
              flush=True)
    print(f"\n=== Top by CDR→epi ===", flush=True)
    for r in sorted(results, key=lambda r: r["cdr_to_epi_min"])[:5]:
        print(f"  {r['name']}: CDR→epi={r['cdr_to_epi_min']:.2f}Å  "
              f"ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  "
              f"H1={r['cdr_seq'][:10]}/H2={r['cdr_seq'][10:16]}/H3={r['cdr_seq'][16:]}",
              flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--snaps", default="/tmp/b5_v13_h3only_snaps.json")
    p.add_argument("--out", default="/tmp/b5_v13_eval.json")
    p.add_argument("--label-prefix", default="v13 step")
    p.add_argument("--init-label", default="v9 step 48 (init)")
    args = p.parse_args()

    snaps, init_seq = load_snaps(args.snaps)
    if not snaps:
        print("No snapshots found!", flush=True)
        sys.exit(1)
    evaluate(snaps, args.out, init_seq, include_init=True,
             label_prefix=args.label_prefix, init_label=args.init_label)
