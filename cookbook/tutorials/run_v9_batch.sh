#!/bin/bash
# Batch v9 design runs — optimal params: wt_logit=2.0, w_prior=0.05, steps=60
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

# ---- 5M2J: TNFα + anti-TNF VHH (3 seeds) ----
for seed in 0 1 2; do
    run_one "TNFA_5M2J" "../../test/5M2J.pdb" "A" \
        "66,67,68,79,80,81,82,83,84,117,118" \
        "antitnf" "" "$seed"
done

# ---- 5M2M: TNFα + VHH3 (3 seeds) ----
for seed in 0 1 2; do
    run_one "TNFA_5M2M" "../../test/5M2M.pdb" "A" \
        "15,16,17,18,58,59,60,61,62,63,99,100,130,131,132,133,135" \
        "vhh3" "" "$seed"
done

# ---- 5JDS: PD-L1 + KN035 (3 seeds, allow CDR Cys for disulfide) ----
for seed in 0 1 2; do
    run_one "PDL1_5JDS" "../../test/5JDS.pdb" "A" \
        "36,38,43,45,48,50,97,98,99,101,102,103,104,105" \
        "kn035" "--allow-cdr-cys" "$seed"
done

echo "===== ALL DONE ====="
