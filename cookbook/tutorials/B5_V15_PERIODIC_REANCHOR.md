# v15 Periodic Re-anchor — Test Results

> **Bottom line: periodic re-anchor is also significantly worse than v9
> fixed epitope.** v15's best Full-eval iptm is **0.576**, vs v9 step 48
> baseline **0.660** in the same eval run, and v9's known global best
> (seed 2 step 56, full 15-seed run) **0.717**. v15 ties v14 (best 0.578)
> and does not break the self-reference problem.

---

## What v15 is

EM-style alternating optimization of the epitope:
- **M-step** (chunk_size=4 design steps): standard v9-style `compute_epitope_loss`
  with a *fixed* epitope list.
- **E-step** (every 4 steps): re-extract the predicted interface from the
  distogram of the current step. For each target residue j, compute
  `min_i E[d(i, j)]` from the distogram. Target residues within
  `epi_threshold=10.0 Å` of any binder residue become the new epitope
  list. Capped to [4, 40] residues.
- The E-step runs in `torch.no_grad()`; it does not disturb the M-step
  gradient.

Initial epitope = v9's 21-residue input-PDB list (preserves v9's anchor
for chunk 0).

## What was tested

| Run | Setup |
|-----|-------|
| Smoke test | steps=4, seed=99 (verified pipeline) |
| Multi-start | seeds 1-5, steps=60, lr=0.05, chunk=4, threshold=10.0 Å |
| Eval | All unique sequences re-folded with Full ESMFold2 (1.3G, 3 loops, 14 samples) |

Per-seed Fast-model results (from the design logs):

| Seed | best CDR→epi (Fast) | best by total (CDR) | best by iptm Fast (CDR) | reanchors |
|------|---------------------|---------------------|--------------------------|-----------|
| 1 | 8.65 Å @ step 45 | `GLQVGYGWGMSYSGQSRVVTDSYQPIYKAPIR` (5 diff) | init (0 diff) | 15 |
| 2 | 8.62 Å @ step 43 | init (0 diff) | `GLQIGYGWAMSYSGQKRVVTDSSQPIVKAPWR` (5 diff) | 15 |
| 3 | 8.96 Å @ step 38 | init (0 diff) | init (0 diff) | 15 |
| 4 | 8.63 Å @ step 47 | `GLQTGYGVYMSYSGQSRVVTDSMQPIYKAPQR` (4 diff) | `GLQPGYGVYMSYSGQSRVVTDSMQPIYKAPQR` (4 diff) | 15 |
| 5 | 8.99 Å @ step 50 | init (0 diff) | init (0 diff) | 15 |

Note: best CDR→epi (Fast) is 8.6-9.0 Å across all 5 seeds — comparable
to v9's 9.52 Å. The optimizer IS finding low CDR→epitope distances, but
this doesn't translate to high Full iptm.

## Full-eval (ground truth) results

| Rank | Name | iptm | pTM | CDR→epi | CDR (32 aa) |
|------|------|------|-----|---------|-------------|
| 1 | v9 step 48 (baseline) | **0.660** | 0.842 | 10.09 | `GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR` |
| 2 | b5_v15_seed3 step 48 | 0.576 | 0.812 | 10.52 | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPGR` |
| 3 | b5_v15_seed1 init | 0.565 | 0.809 | 10.68 | init (0 diff) |
| 4 | b5_v15_seed5 step 52 | 0.537 | 0.798 | 10.94 | `GLQIGYGWYMSYSGQSRVVTPSPQPIYKAPIR` |
| 5 | b5_v15_seed5 step 48 | 0.503 | 0.785 | 11.39 | `GLQIGYGNYMSYSGQSRVVTDSPQPIYKAPIR` |
| 6 | b5_v15_seed3 step 44 | 0.499 | 0.785 | 11.25 | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPMR` |
| 7 | b5_v15_seed4 step 40 | 0.452 | 0.767 | 11.76 | `GLQIGYGHYMSYSGQSRVVTDSYQPIYKAPIR` |

29 unique sequences, max iptm = 0.660 (v9 baseline), 0.576 (best v15).

## Comparison to v9 / v14

| Version | Best iptm (Full) | Best CDR→epi (Full) | Best CDR (32 aa) | Self-ref fixed? |
|---------|------------------|---------------------|-------------------|------------------|
| **v9** (15 seeds) | **0.717** | 9.52 Å | `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR` (5 diff) | n/a (external anchor) |
| v9 step 48 (baseline in this eval) | 0.660 | 10.09 | `GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR` | n/a |
| v14 topk (5 seeds) | 0.578 | n/a | init (0 diff) | **NO** — every step |
| **v15 re-anchor (5 seeds)** | **0.576** | 10.46 | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPGR` (2 diff) | **NO** — every 4 steps |

## Why v15 failed (and why it's not v9's fault)

The hypothesis was: "every chunk, the epi is fixed, so the gradient is
informative." The result shows: even at chunk_size=4, the epi still
moves enough to defeat this. The re-anchor log on seed 3 shows the
epi flipping by ±5-8 residues every chunk:

```
step  4  |epi| 21->26  + 8 - 3
step  8  |epi| 26->23  + 0 - 3
step 12  |epi| 23->28  + 6 - 1
step 20  |epi| 31->23  + 0 - 8     <-- 8 residues dropped at once
step 32  |epi| 32->27  + 0 - 5
step 60  |epi| 30->27  + 3 - 6
```

The fundamental issue: the epi is **derived from the predicted
distogram of the current state**. So at any chunk boundary, the new
"target" is "wherever I am right now, plus a 10 Å neighborhood." The
optimizer gets pulled toward the moving target but the target always
follows the optimizer — net effect, the optimizer meanders in a
self-consistent way without ever settling in a basin.

Concretely:
- v15 best by total (Fast) is mostly the init (0 diffs) — the
  optimizer didn't move the CDRs because there was no consistent
  signal to move them.
- v15 best by iptm (Full) has 1-2 CDR diffs from init. These are
  small perturbations near the init basin, not the new basin that v9
  found (5 diffs, 0.717).

## What v15's behavior does suggest

1. **The re-anchor is happening as designed**: epitope size fluctuates
   15-36 across chunks (vs 21 fixed in v9). The mechanism works.
2. **The optimizer IS finding low CDR→epitope distances (8.6-9.0 Å)**,
   but the Full-eval iptm stays in the 0.4-0.6 range, not the 0.7+ of
   v9. The min-distance metric is decoupled from binding quality.
3. **H1-8 V→W recurs** in 3 of the top 7 v15 results (`GLQ...YGW...`).
   This is the same H1-8 V→M (or W) substitution that v9 multi-start
   independently discovered 5 times in 15 seeds. It's a robust
   signal in the loss landscape.

## Conclusion

The fundamental problem isn't "epi is fixed in v9 vs dynamic in
v14/v15." The fundamental problem is: **the epitope loss must come
from a target that's external to the current optimization state.**

- v9: external target = input PDB (held fixed)  → works (0.717)
- v14: internal target = topk of current distogram (per step) → fails
- v15: internal target = threshold of current distogram (per chunk) → fails

Going forward, the only meaningful "dynamic epitope" experiment left
is one where the new target comes from a *different* source than the
current distogram. E.g.:
- Predict structure, then **fold with target-only sequence and find
  the surface accessible to antibody-style H3** (true external
  surface).
- Use a *frozen-in-time* prior from a fold done N steps ago, then
  re-extract the interface from THAT prior.
- Use the input-PDB epitope as a hard core, but allow the epi to grow
  outward by 1-2 residues per chunk.

These would actually be "external-to-current-state" — a real change.
v15 is not that; it's a slower version of v14's self-reference.

## Files

- `design_b5_mps_v15_periodic_reanchor.py` — the new design script.
- `run_v15_periodic_reanchor.sh` — 5-seed shell script.
- `/tmp/b5_v15_seed{1..5}.log` — per-seed design loop logs.
- `/tmp/b5_v15_seed{1..5}_snaps.json` — per-seed snapshots with reanchor_log.
- `/tmp/b5_v15_multistart_eval.json` — Full-eval of 29 unique sequences.
- `/tmp/b5_v15_multistart_eval.log` — Full-eval printout.
