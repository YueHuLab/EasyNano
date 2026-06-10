#!/bin/bash
# Second batch: Ty1/RBD + 6WAQ/RBD/VHH72 — optimal params
set -e
cd /Users/huyue/esmc_design_new/esm-main-2/cookbook/tutorials
OUT="/tmp/v9_designs"
mkdir -p "$OUT"

run_one() {
    local tag=$1 pdb=$2 tchain=$3 epi=$4 framework=$5 extra_args=$6 seed=$7
    local name="${tag}_seed${seed}"
    echo "===== $(date) $name ====="
    python3 design_target_v9.py \
        --target-pdb "$pdb" --target-chain "$tchain" \
        --epitope-indices "$epi" --framework "$framework" \
        --steps 60 --lr 0.05 --wt-logit 2.0 --w-prior 0.05 \
        --seed "$seed" \
        --snapshot-path "$OUT/${name}_snaps.json" \
        $extra_args \
        2>&1 | tee "$OUT/${name}.log"
    echo "===== $(date) $name DONE ====="
}

# ---- Ty1/RBD (extracted RBD from 6ZXN, 3 seeds) ----
for seed in 0 1 2; do
    run_one "RBD_6ZXN_TY1" "/tmp/6ZXN_RBD.pdb" "A" \
        "18,19,22,116,117,118,119,120,122,138,140,152,153,154,155,156,159,160,162,163,164" \
        "ty1" "" "$seed"
done

# ---- RBD/VHH72 (6WAQ, 3 seeds) ----
for seed in 0 1 2; do
    run_one "RBD_6WAQ_VHH72" "../../test/6WAQ.pdb" "B" \
        "35,36,37,38,39,40,41,42,43,44,45,46,49,50" \
        "vhh72" "" "$seed"
done

echo "===== ALL DONE ====="
