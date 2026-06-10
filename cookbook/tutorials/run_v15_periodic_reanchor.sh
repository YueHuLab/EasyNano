#!/usr/bin/env bash
# v15 multi-start: 5 seeds (1-5) of v15 with periodic epitope re-anchoring.
# Direct comparison to v9 (fixed) and v14 (topk dynamic).
# Each run: load Full model for prior (~30s), then 60 design steps with Fast model (~14 min on MPS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for seed in 1 2 3 4 5; do
  echo "=== v15 periodic-reanchor multi-start seed=$seed ==="
  python3 -u design_b5_mps_v15_periodic_reanchor.py \
    --steps 60 --lr 0.05 --seed $seed \
    --chunk-size 4 --epi-threshold 10.0 \
    --snapshot-path /tmp/b5_v15_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v15_seed${seed}.log 2>&1
  echo "  seed=$seed done; $(grep -c '^  ' /tmp/b5_v15_seed${seed}.log) lines"
done
echo "=== all v15 multi-start runs complete ==="

# === Eval all 5 seeds with Full ===
echo "=== Eval v15 seeds 1-5 with Full ==="
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v15_seed1_snaps.json /tmp/b5_v15_seed2_snaps.json \
          /tmp/b5_v15_seed3_snaps.json /tmp/b5_v15_seed4_snaps.json \
          /tmp/b5_v15_seed5_snaps.json \
  --out /tmp/b5_v15_multistart_eval.json \
  > /tmp/b5_v15_multistart_eval.log 2>&1
echo "  v15 multi-start eval done"
