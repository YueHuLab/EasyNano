#!/usr/bin/env bash
# Multi-start v9 expanded: seeds 6-15 (10 more seeds).
# 5 seeds (1-5) already done; seed=2 found the global-best basin (iptm=0.717).
# 10 more seeds tests whether even better basins exist.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for seed in 6 7 8 9 10 11 12 13 14 15; do
  echo "=== v9 multi-start seed=$seed ==="
  python3 -u design_b5_mps_v9_cacoord.py \
    --steps 60 --lr 0.05 --seed $seed \
    --snapshot-path /tmp/b5_v9_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v9_seed${seed}.log 2>&1
  echo "  seed=$seed done; $(grep -c '^  ' /tmp/b5_v9_seed${seed}.log) lines"
done
echo "=== expanded multi-start runs complete ==="

# === Eval all 10 new seeds (6-15) with Full model ===
echo "=== Eval seeds 6-15 with Full ==="
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v9_seed6_snaps.json /tmp/b5_v9_seed7_snaps.json \
          /tmp/b5_v9_seed8_snaps.json /tmp/b5_v9_seed9_snaps.json \
          /tmp/b5_v9_seed10_snaps.json /tmp/b5_v9_seed11_snaps.json \
          /tmp/b5_v9_seed12_snaps.json /tmp/b5_v9_seed13_snaps.json \
          /tmp/b5_v9_seed14_snaps.json /tmp/b5_v9_seed15_snaps.json \
  --out /tmp/b5_v9_multistart_expanded_eval.json \
  > /tmp/b5_v9_multistart_expanded_eval.log 2>&1
echo "  expanded multi-start eval done"
