"""Step 4: Iterative v9 design starting from v9_best_15seed_p116Y init.

After Steps 1-3 we found that v9_best_15seed_p116Y (CDR =
GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY, with R→Y at last CDR H3 position)
is a robust high-iptm candidate: median 0.692, std 0.020 across 9 samples.

Now run v9 iterative design starting from this seed, with multiple seeds,
and look for even better CDRs in the same high-iptm basin.

The framework is fully pinned; only 32 CDR positions are mutable.

Default: 5 seeds × 60 steps (matches prior v9 runs).
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import json
import subprocess
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

# Init: v9_best_15seed_p116Y (CDR H3 last position R→Y)
INIT_SEQ = ("QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGMYMSLGWFRQAPGQGLEAVAAI"
            "SYSGQKTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSST"
            "PIYKAGIYWGQGTLVTVS")

# Match v9 defaults
STEPS = 30
LR = 0.05
N_SEEDS = 3
SCRIPT = HERE / "design_b5_mps_v9_cacoord.py"


def run_one_seed(seed: int, snapshot_path: Path) -> dict:
    """Run v9 design for one seed, return the snapshot JSON path."""
    print(f"\n{'='*70}", flush=True)
    print(f"=== v9 design seed={seed} ===", flush=True)
    print(f"{'='*70}", flush=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--init-seq", INIT_SEQ,
        "--steps", str(STEPS),
        "--lr", str(LR),
        "--seed", str(seed),
        "--snapshot-path", str(snapshot_path),
        "--log-every", "10",
        "--snapshot-every", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(HERE))
    dt = time.time() - t0
    print(f"\n  seed={seed} done in {dt:.0f}s, exit={proc.returncode}",
          flush=True)
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            snaps = json.load(f)
        return {"seed": seed, "time_s": dt, "snaps": snaps,
                "snapshot_path": str(snapshot_path)}
    return {"seed": seed, "time_s": dt, "snaps": None,
            "snapshot_path": None}


def main():
    out_dir = Path("/tmp/b5_iter_p116Y")
    out_dir.mkdir(exist_ok=True)

    all_results = []
    for s in range(N_SEEDS):
        snap_path = out_dir / f"snaps_seed{s}.json"
        r = run_one_seed(s, snap_path)
        all_results.append(r)

    # Aggregate per-step iptm per seed
    print(f"\n{'='*70}", flush=True)
    print("=== AGGREGATE: per-step iptm evolution ===", flush=True)
    print(f"{'='*70}\n", flush=True)

    for r in all_results:
        if r["snaps"] is None:
            print(f"  seed {r['seed']}: no snapshots", flush=True)
            continue
        snaps = r["snaps"]
        steps = sorted(snaps.keys(), key=lambda x: int(x))
        iptms = [(s, snaps[s].get("iptm")) for s in steps]
        best = max(iptms, key=lambda x: x[1] if x[1] is not None else 0)
        last = iptms[-1]
        print(f"  seed {r['seed']}:  first_iptm={iptms[0][1]:.3f}  "
              f"best_iptm={best[1]:.3f}@{best[0]}  "
              f"last_iptm={last[1]:.3f}@{last[0]}", flush=True)

    # Save the per-seed best sequences and their iptm
    out = {
        "init_seq": INIT_SEQ,
        "n_seeds": N_SEEDS,
        "steps": STEPS,
        "lr": LR,
        "results": [
            {"seed": r["seed"], "time_s": r["time_s"],
             "snapshot_path": r["snapshot_path"]}
            for r in all_results
        ],
    }
    with open(out_dir / "iter_p116Y.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/iter_p116Y.json", flush=True)


if __name__ == "__main__":
    main()
