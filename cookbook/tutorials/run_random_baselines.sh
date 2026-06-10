#!/bin/bash
# Sequential random baselines for all 5 targets
cd /Users/huyue/esmc_design_new/esm-main-2/cookbook/tutorials
OUT="/tmp/v9_designs"

# Ty1/RBD (best result)
echo "=== Ty1/RBD random baseline ==="
python3 random_baseline.py \
    --target-pdb /tmp/6ZXN_RBD.pdb --target-chain A \
    --epitope-indices "18,19,22,116,117,118,119,120,122,138,140,152,153,154,155,156,159,160,162,163,164" \
    --framework ty1 --n-random 30 --seed 42 \
    --out $OUT/RBD_6ZXN_TY1_random.json 2>&1 | tail -5

# PD-L1/KN035
echo "=== PD-L1/KN035 random baseline ==="
python3 random_baseline.py \
    --target-pdb ../../test/5JDS.pdb --target-chain A \
    --epitope-indices "36,38,43,45,48,50,97,98,99,101,102,103,104,105" \
    --framework kn035 --n-random 30 --seed 42 \
    --out $OUT/PDL1_5JDS_random.json 2>&1 | tail -5

# TNFα/VHH3
echo "=== TNFα/VHH3 random baseline ==="
python3 random_baseline.py \
    --target-pdb ../../test/5M2M.pdb --target-chain A \
    --epitope-indices "15,16,17,18,58,59,60,61,62,63,99,100,130,131,132,133,135" \
    --framework vhh3 --n-random 30 --seed 42 \
    --out $OUT/TNFA_5M2M_random.json 2>&1 | tail -5

# TNFα/anti-TNF
echo "=== TNFα/anti-TNF random baseline ==="
python3 random_baseline.py \
    --target-pdb ../../test/5M2J.pdb --target-chain A \
    --epitope-indices "66,67,68,79,80,81,82,83,84,117,118" \
    --framework antitnf --n-random 30 --seed 42 \
    --out $OUT/TNFA_5M2J_random.json 2>&1 | tail -5

# RBD/VHH72
echo "=== RBD/VHH72 random baseline ==="
python3 random_baseline.py \
    --target-pdb ../../test/6WAQ.pdb --target-chain B \
    --epitope-indices "35,36,37,38,39,40,41,42,43,44,45,46,49,50" \
    --framework vhh72 --n-random 30 --seed 42 \
    --out $OUT/RBD_6WAQ_VHH72_random.json 2>&1 | tail -5

echo "=== ALL RANDOM BASELINES DONE ==="
