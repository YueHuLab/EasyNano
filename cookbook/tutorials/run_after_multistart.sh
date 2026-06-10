#!/usr/bin/env bash
# Run after multi-start completes:
# 1. v13 (H3-only, 14 min)
# 2. v12 (Full in loop, 30-60 min)
# 3. Eval all results
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# === Step 1: v13 H3-only ===
echo "=== v13 H3-only from v9 step 48 ==="
python3 -u design_b5_mps_v13_h3only.py \
  --steps 60 --lr 0.05 --seed 0 \
  --snapshot-path /tmp/b5_v13_h3only_snaps.json \
  --log-every 4 --snapshot-every 4 \
  --full-loops 3 --full-samples 14 --full-diffusion 4 \
  > /tmp/b5_v13_h3only.log 2>&1
echo "  v13 done"

# === Step 2: v12 Full in loop (slow) ===
echo "=== v12 Full ESMFold2 in design loop ==="
python3 -u design_b5_mps_v12_full_in_loop.py \
  --steps 30 --lr 0.05 --seed 0 --sample-steps 10 \
  --snapshot-path /tmp/b5_v12_full_in_loop_snaps.json \
  --log-every 2 --snapshot-every 4 \
  --full-loops 3 --full-samples 14 --full-diffusion 4 \
  > /tmp/b5_v12_full_in_loop.log 2>&1
echo "  v12 done"

# === Step 3: Eval ===
echo "=== Eval v12 with Full ==="
python3 -u eval_v12_candidates.py \
  --snaps /tmp/b5_v12_full_in_loop_snaps.json \
  --out /tmp/b5_v12_eval.json \
  > /tmp/b5_v12_eval.log 2>&1
echo "  v12 eval done"

echo "=== Eval v13 with Full ==="
python3 -u eval_v13_candidates.py \
  --snaps /tmp/b5_v13_h3only_snaps.json \
  --out /tmp/b5_v13_eval.json \
  > /tmp/b5_v13_eval.log 2>&1
echo "  v13 eval done"

echo "=== Eval multi-start v9 seeds 1-5 with Full ==="
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v9_seed1_snaps.json /tmp/b5_v9_seed2_snaps.json \
          /tmp/b5_v9_seed3_snaps.json /tmp/b5_v9_seed4_snaps.json \
          /tmp/b5_v9_seed5_snaps.json \
  --out /tmp/b5_v9_multistart_eval.json \
  > /tmp/b5_v9_multistart_eval.log 2>&1
echo "  multi-start eval done"

echo "=== All done ==="
