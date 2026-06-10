#!/usr/bin/env bash
# Multi-start v9: run seeds 1-5 in series, save snapshots to per-seed files.
# Each run: load Full model for prior (one-time per run, ~30s),
#           then 60 design steps with Fast model (~14 min on MPS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for seed in 1 2 3 4 5; do
  echo "=== v9 multi-start seed=$seed ==="
  python3 -u design_b5_mps_v9_cacoord.py \
    --steps 60 --lr 0.05 --seed $seed \
    --snapshot-path /tmp/b5_v9_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v9_seed${seed}.log 2>&1
  echo "  seed=$seed done; $(grep -c '^  ' /tmp/b5_v9_seed${seed}.log) lines"
done
echo "=== all multi-start runs complete ==="
