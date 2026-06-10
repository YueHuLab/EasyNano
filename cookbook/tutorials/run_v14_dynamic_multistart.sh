#!/usr/bin/env bash
# v14 multi-start: 5 seeds (1-5) of v14 with --epitope-mode topk (dynamic).
# Direct comparison to v9 seeds 1-5 (which used fixed epitope).
# Each run: load Full model for prior (~30s), then 60 design steps with Fast model (~14 min on MPS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for seed in 1 2 3 4 5; do
  echo "=== v14 dynamic-epitope multi-start seed=$seed ==="
  python3 -u design_b5_mps_v14_dynamic_epitope.py \
    --steps 60 --lr 0.05 --seed $seed \
    --epitope-mode topk --topk-k 8 \
    --snapshot-path /tmp/b5_v14_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v14_seed${seed}.log 2>&1
  echo "  seed=$seed done; $(grep -c '^  ' /tmp/b5_v14_seed${seed}.log) lines"
done
echo "=== all v14 multi-start runs complete ==="

# === Eval all 5 seeds with Full model ===
echo "=== Eval v14 seeds 1-5 with Full ==="
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v14_seed1_snaps.json /tmp/b5_v14_seed2_snaps.json \
          /tmp/b5_v14_seed3_snaps.json /tmp/b5_v14_seed4_snaps.json \
          /tmp/b5_v14_seed5_snaps.json \
  --out /tmp/b5_v14_multistart_eval.json \
  > /tmp/b5_v14_multistart_eval.log 2>&1
echo "  v14 multi-start eval done"
