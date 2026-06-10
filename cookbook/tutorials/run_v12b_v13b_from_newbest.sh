#!/usr/bin/env bash
# Run v12b (Full in loop) and v13b (H3-only) BOTH starting from the new best
# (v9 multi-start seed=2 step=56, iptm=0.717). v12/v13 starting from v9 step 48
# both re-converged to the v9 basin; starting from the new basin tests whether
# Full-in-loop or H3-only restriction can refine or escape from this different
# starting point.
#
# New best full binder:
# QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGMYMSLGWFRQAPGQGLEAVAAISYSGQKTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSSTPIYKAGIRWGQGTLVTVS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

NEW_BEST="QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGMYMSLGWFRQAPGQGLEAVAAISYSGQKTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSSTPIYKAGIRWGQGTLVTVS"

# === Step 1: v13b H3-only from new best ===
echo "=== v13b: H3-only from new best (seed=2 step 56) ==="
python3 -u design_b5_mps_v13_h3only.py \
  --steps 60 --lr 0.05 --seed 0 \
  --init-seq "$NEW_BEST" \
  --snapshot-path /tmp/b5_v13b_h3only_snaps.json \
  --log-every 4 --snapshot-every 4 \
  --full-loops 3 --full-samples 14 --full-diffusion 4 \
  > /tmp/b5_v13b_h3only.log 2>&1
echo "  v13b done"

# === Step 2: v12b Full in loop from new best ===
echo "=== v12b: Full ESMFold2 in design loop, from new best ==="
python3 -u design_b5_mps_v12_full_in_loop.py \
  --steps 30 --lr 0.05 --seed 0 --sample-steps 10 \
  --init-seq "$NEW_BEST" \
  --snapshot-path /tmp/b5_v12b_full_in_loop_snaps.json \
  --log-every 2 --snapshot-every 4 \
  --full-loops 3 --full-samples 14 --full-diffusion 4 \
  > /tmp/b5_v12b_full_in_loop.log 2>&1
echo "  v12b done"

# === Step 3: Eval v12b with Full ===
echo "=== Eval v12b with Full ==="
python3 -u eval_v12_candidates.py \
  --snaps /tmp/b5_v12b_full_in_loop_snaps.json \
  --out /tmp/b5_v12b_eval.json \
  --label-prefix "v12b step" \
  --init-label "seed=2 step 56 (init, new best)" \
  > /tmp/b5_v12b_eval.log 2>&1
echo "  v12b eval done"

# === Step 4: Eval v13b with Full ===
echo "=== Eval v13b with Full ==="
python3 -u eval_v13_candidates.py \
  --snaps /tmp/b5_v13b_h3only_snaps.json \
  --out /tmp/b5_v13b_eval.json \
  --label-prefix "v13b step" \
  --init-label "seed=2 step 56 (init, new best)" \
  > /tmp/b5_v13b_eval.log 2>&1
echo "  v13b eval done"

echo "=== All v12b/v13b done ==="
