"""v10a: Iter round 2 — fold(v9 step 48) prior, design from v9 step 48.

Plan B+ iteration, single-variable test: does iterating structure-conditional
optimization help when v9 already found a new optimum (iptm=0.661)?

Compared to v9:
  - Starting point: v9 step 48 (best so far)  vs  v2 step050
  - Prior source:  fold(v9 step 48)  vs  fold(v2 step050)
  - Everything else: identical (lr=0.05, steps=60, single Fast model, no PLM)

If v10a finds a strictly better iptm, structure-conditional iteration is
validated. If not, we have a stable local optimum and should try v10b's
exploration changes (cosine T + lr(T)) to escape.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from design_b5_mps_v9_cacoord import run_design, predict_prior_from_full_ca  # noqa: E402
from test_b5_pdb import setup_design  # noqa: E402

# v9 step 48 full sequence — best so far (iptm=0.661, pTM=0.843, CDR→epi=10.02 Å)
V9_STEP48_FULL = (
    "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGWYMSLGWFRQAPGQGLEAVAAI"
    "SYSGQRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVADSPQR"
    "IYKAPIRWGQGTLVTVS"
)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wt-logit", type=float, default=5.0)
    p.add_argument("--w-epitope", type=float, default=0.2)
    p.add_argument("--w-prior", type=float, default=0.3)
    p.add_argument("--full-loops", type=int, default=3)
    p.add_argument("--full-samples", type=int, default=14)
    p.add_argument("--full-diffusion", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_v10a_iter_snaps.json")
    p.add_argument("--init-seq", type=str, default=V9_STEP48_FULL)
    p.add_argument("--use-wt-prior", action="store_true")
    args = p.parse_args()

    print(f"=== v10a: iter round 2, starting from v9 step 48 ===", flush=True)
    print(f"  init CDR: H1={args.init_seq[25:35]}  "
          f"H2={args.init_seq[54:60]}  H3={args.init_seq[101:117]}", flush=True)

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
