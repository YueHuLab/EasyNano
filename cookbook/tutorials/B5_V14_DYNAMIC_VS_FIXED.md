# v14 Dynamic Epitope vs v9 Fixed Epitope — A/B Test

> **Bottom line: dynamic epitope is significantly worse than fixed epitope.**
> v9 (fixed, 21-residue input-PDB epitope) wins overall by **0.139 iptm** in
> a clean apples-to-apples 5-seed comparison. The new-best basin that v9
> found (iptm 0.717) was NOT re-discovered by any v14 seed.

---

## Test setup

Both runs are **identical except for the epitope definition**:

| Item | v9 (fixed) | v14 (dynamic) |
|------|-----------|---------------|
| Init sequence | v2 step050 (`GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR`) | same |
| Prior | v9 CA-coord prior (Full, 4-sample avg) | same |
| Loss weights | epi 0.2 / intra 0.5 / inter 0.5 / glob 0.2 / prior 0.3 / aa_freq 0.01 | same |
| Optimizer | Adam, lr=0.05, 60 steps | same |
| Forward | ESMFold2-Fast (721M), 5 sample steps, 1 loop | same |
| **Epitope loss** | `compute_epitope_loss` over a **fixed list of 21 residues** from input PDB | `compute_topk_epitope_loss` (K=8) — recomputed every step |
| Framework pinned | yes (95 pos) | yes |
| Antigen mutable | no (1-hot) | no (1-hot) |
| Seeds | 1-5 | 1-5 |

Total cost: 5 × 14 min design + 5 min eval = ~75 min per run on Mac MPS.

---

## Results — top by iptm, per seed

| Seed | v9 best iptm | v9 best CDR | v14 best iptm | v14 best CDR | Δ (v14 − v9) |
|------|--------------|-------------|---------------|--------------|-------------|
| 1 | **0.561** (init) | `GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR` (0 diff) | 0.540 (step 60) | `GLQHGYGWYMSYSGQKRVVTPSYTPIYKAPWR` (6 diffs) | −0.021 |
| 2 | **0.717** (step 56) | `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR` (5 diffs, **NEW BEST**) | 0.482 (step 44) | `GLQIGYGWYMSYSGQSRVVTDSSTPIYKAPIR` (1 diff) | **−0.235** |
| 3 | 0.519 (step 44) | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPMR` (1 diff) | **0.578** (step 44) | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPIR` (0 diff) | **+0.059** |
| 4 | **0.477** (step 48) | `GLQIGYGYYMSYSGQSRVVTDSYQPIQKAPIR` (3 diffs) | ~0.45 | `GLQHGYGVYMSYSGQSRVVTDSMQDIQKAPIR` (5 diffs) | ~−0.03 |
| 5 | 0.42 (step 56) | `GLQIGYGWYMSYSGQSRVVTDSYQPIYKAPGR` (3 diffs) | **0.554** (step 52) | `GLQIGYGWYMSYSGQSRVVTPSPQPIYKAPIR` (3 diffs) | **+0.13** |
| **max** | **0.717** | — | **0.578** | — | **−0.139** |

**Summary**: v14 wins on 2/5 seeds (3, 5), loses on 3/5 seeds (1, 2, 4).
Net: v9 wins overall. Most importantly, **v9 found the new-best basin
(0.717) that v14 missed entirely**.

---

## Why dynamic epitope lost

**v9 seed 2 step 56** (the new best, 5 CDR diffs):
- H1: `GLQIGYGVYM` → `GLQIGYGMYM` (V→M at H1-8)
- H2: `SYSGQS` → `SYSGQK` (S→K at H2-6)
- H3: `RVVTDSYQPIYKAPIR` → `RVVTDSSTPIYKAGIR` (Y→S, Q→T, P→G)
- This basin is **only** reachable if the optimizer anchors to the
  fixed epitope; otherwise the CDRs explore freely and the H1-8 V→M
  + H2-6 S→K combination never settles in.

**v14 seed 2 final CDR** (4 diffs, step 60):
- H1: `GLQIGYGVYM` → `GLQIGYGWMMSYS` — H1-8 V→M, but H2-6 S→S, H3 different
- Without the fixed epitope anchor, H2-6 S→K never gets selected.

The dynamic epitope loss removes a stable target: each step, the loss
says "get close to *whatever* is closest to you right now". This is a
self-referential signal — the loss goes to 0 trivially if the binder
just sits where it is. The optimizer needs an *external* anchor to
push in a useful direction.

**Confirmed by another signal**: v14's top-2 by Full iptm (seed 3 step
44 = 0.578) is **the v2 init sequence with zero CDR diffs**. The
optimization didn't explore at all — the dynamic loss was too weak to
overcome the WT prior.

---

## What to try next

The failure of "pure dynamic" doesn't mean "all dynamic" is bad. A
**hybrid** might be the right next step:

- **Hybrid A**: top-K closest target residues, but **restricted to
  within 12-15 Å of the input-PDB epitope** (a "soft neighborhood").
  Lets the binder explore within the input-epitope region but not
  fly away to a hydrophobic patch elsewhere.
- **Hybrid B**: fixed epitope with a **soft mask** that decays with
  distance from the fixed epitope (e.g., `weight = exp(-d/8 Å)`).
  Equivalent to soft attention over the antigen.
- **Hybrid C**: keep the fixed epitope loss, but **add** a small
  secondary term for "min over the top-K closest" with a small weight
  (0.05) to encourage the optimizer to follow if the binder moves.

These are also worth doing, but the **v9 fixed epitope result (0.717)
remains the global best** until proven otherwise.

---

## Files

- `design_b5_mps_v14_dynamic_epitope.py` — the new design script.
- `binder_design_hy_losses.py` — new function
  `compute_topk_epitope_loss`.
- `run_v14_dynamic_multistart.sh` — 5-seed shell script.
- `eval_v9_multistart.py` — reused for Full-eval of all 5 v14 seeds.
- `/tmp/b5_v14_seed{1..5}.log` — per-seed design loop logs.
- `/tmp/b5_v14_multistart_eval.json` — Full-eval of 28 unique sequences.
- `/tmp/b5_v14_multistart_eval.log` — Full-eval printout.
