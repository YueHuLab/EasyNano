#!/usr/bin/env bash
# v16 multi-start: 5 seeds (1-5) of v16 with one-time Full-fold re-anchor at step 30.
# Direct comparison to v9 (fixed) and v14 (topk) and v15 (re-anchor every 4).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for seed in 1 2 3 4 5; do
  echo "=== v16 fullfold-reanchor multi-start seed=$seed ==="
  python3 -u design_b5_mps_v16_fullfold_reanchor.py \
    --steps 60 --lr 0.05 --seed $seed \
    --reanchor-step 30 --full-epi-threshold 10.0 \
    --snapshot-path /tmp/b5_v16_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v16_seed${seed}.log 2>&1
  echo "  seed=$seed done; $(grep -c '^  ' /tmp/b5_v16_seed${seed}.log) lines"
done
echo "=== all v16 multi-start runs complete ==="

# === Eval all 5 seeds with Full ===
echo "=== Eval v16 seeds 1-5 with Full ==="
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v16_seed1_snaps.json /tmp/b5_v16_seed2_snaps.json \
          /tmp/b5_v16_seed3_snaps.json /tmp/b5_v16_seed4_snaps.json \
          /tmp/b5_v16_seed5_snaps.json \
  --out /tmp/b5_v16_multistart_eval.json \
  > /tmp/b5_v16_multistart_eval.log 2>&1
echo "  v16 multi-start eval done"
