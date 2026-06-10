#!/bin/bash
# Quick-validation panel: 3 targets × 5 seeds × 100 steps.
# Each target uses its own WT VHH framework ("self-recovery" test).
#
# Targets:
#   1. PD-L1 (5JDS) + KN035 framework  (--allow-cdr-cys for H3 disulfide)
#   2. RBD   (6WAQ) + VHH-72 framework
#   3. TNFα  (5M2J) + anti-TNF VHH framework
#
# Total: 15 design runs × ~10 min each = ~2.5 hours
set -e

mkdir -p /tmp/quick_val/runs

# Indexed arrays (macOS bash compat)
TAGS=("PDL1_5JDS_KN035" "RBD_6WAQ_VHH72" "TNFA_5M2J_ANTITNF")
PDBS=("test/5JDS.pdb"    "test/6WAQ.pdb"   "test/5M2J.pdb")
CHAINS=("A"              "B"               "A")
EPIS=("36,38,43,45,48,50,97,98,99,101,102,103,104,105" \
      "35,36,37,38,39,40,41,42,43,44,45,46,49,50" \
      "66,67,68,79,80,81,82,83,84,117,118")
FWS=("kn035" "vhh72" "antitnf")
EXTRAS=("--allow-cdr-cys" "" "")

for i in 0 1 2; do
    tag="${TAGS[$i]}"
    pdb="${PDBS[$i]}"
    chain="${CHAINS[$i]}"
    epi="${EPIS[$i]}"
    fw="${FWS[$i]}"
    extra="${EXTRAS[$i]}"
    echo ""
    echo "=========================================="
    echo "TARGET [$i]: $tag"
    echo "  PDB: $pdb, chain $chain"
    echo "  Epitope: $epi"
    echo "  Framework: $fw $extra"
    echo "=========================================="
    for seed in 0 1 2 3 4; do
        snap_path="/tmp/quick_val/runs/${tag}_seed${seed}.json"
        if [ -f "$snap_path" ] && [ -s "$snap_path" ]; then
            echo "  [SKIP] $snap_path already exists"
            continue
        fi
        echo ""
        echo "--- $tag seed $seed ---"
        python3 cookbook/tutorials/design_target.py \
            --target-pdb "$pdb" --target-chain "$chain" \
            --epitope-indices "$epi" \
            --framework "$fw" --seed $seed --steps 100 --log-every 25 \
            --snapshot-path "$snap_path" $extra 2>&1 | tail -8
    done
done

echo ""
echo "=========================================="
echo "All design runs complete."
echo "=========================================="
