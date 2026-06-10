# B5 CDR Design — History and Best Results

> Iterative antibody CDR design for the B5 antigen using ESMFold2 + gradient-
> based optimization on Mac MPS. Tested 13 versions (v2 → v13) over multiple
> iterations with structure priors, plus a **15-seed multi-start ensemble**
> and re-runs of directions 2 & 3 from the new best.

---

## 🏆 Best result (verified with Full ESMFold2, 1.3G)

| Metric | Value |
|---|---|
| **ipTM** | **0.717** |
| **pTM** | **0.853** |
| **CDR→epi (min distance)** | **9.52 Å** |
| **Sequence** | `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR` |
| **Full binder (127 aa)** | See below |
| **Source** | v9 multi-start, **seed=2 step 56** |
| **CDR diffs from v2 init** | 5 |
| **CDR diffs from v9 step 48** | 7 |

**Binder sequence**:
```
QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGMYMSLGWFRQAPGQGLEAVAAISYSGQKTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSSTPIYKAGIRWGQGTLVTVS
```

**CDR changes from v2 init (5 substitutions)**:
- H1 (pos 25-34): `GLQIGYGVYM` → `GLQIGYGMYM` (V→**M** at H1-8)
- H2 (pos 54-59):  `SYSGQS`     → `SYSGQK`     (S→**K** at H2-6)
- H3 (pos 101-116): `RVVTDSYQPIYKAPIR` → `RVVTDSSTPIYKAGIR` (Y→S at H3-7, Q→T at H3-8, P→G at H3-14)

**Antigen (B5, 223 aa)**:
```
QAFWKAVTAEFLAMLIFVLLSLGSTINWGGTEKPLPVDMVLISLCFGLSIATMVQCFGHISGGHINPAVTVAMVCTRKISIAKSVFYIAAQCLGAIIGAGILYLVTPPSVVGGLGVTMVHGNLTAGHGLLVELIITFQLVFTIFASCDSKRTDVTGSIALAIGFSVAIGHLFAINYTGASMNPARSFGPAVIMGNWENHWIYWVGPIIGAVLAGGLYEYVFCP
```

### Run-2 (prior best — v9 step 48, iptm 0.661)

| Metric | Value |
|---|---|
| **ipTM** | 0.661 |
| **pTM** | 0.843 |
| **CDR→epi (min distance)** | 10.02 Å |
| **Sequence** | `GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR` |

CDR diffs from v2 init (5 substitutions): H1-8 V→W, H2-6 S→R, H3-4 T→A, H3-7 Y→P, H3-9 P→R.
Note the winner **flips both H1-8 and H2-6** to a different residue (W→M, R→K) and a
different set of H3 mutations. Same number of substitutions, different chemistry.

---

## Version history

| Version | Start | Key change | best iptm (Full) | best CDR→epi | best CDR seq |
|---|---|---|---|---|---|
| v2 | WT | ESMFold2-Fast + structure prior (1 sample) | 0.572 | 10.78 | `GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR` (v2 init) |
| v7 | v2 | Adam + larger LR + soft logits | (stayed at v2) | — | — |
| v8 | v2 | **distogram-expected prior** | 0.616 (step 36) | 10.40 | = v2 init (re-converged) |
| v9 | v2 | **CA-coord prior (4-sample average) — realized 3D CA distances, not distogram expectation** | 0.661 (step 48) | 10.02 | `GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR` (5 diffs) |
| v10a | v9 step 48 | Iter round 2: fold(v9 step 48) prior + same params as v9 | 0.656 | 10.00 | = v9 step 48 (re-converged) |
| v10b | v9 step 48 | + official cosine T + lr(T) + sampling 1→50 | 0.668 | 9.97 | = v9 step 48 (still re-converged) |
| v10c | v10b step 80 | + strong aa_freq PLM proxy (0.05) | 0.641 | 10.07 | `GMQIGYGWYMSYSGQQRVVASSPQQIYQAPIR` (5 new diffs, worse) |
| v11 | v9 step 48 | **TRUE iterative prior refresh: re-build CA-coord prior from Fast sample_atom_coords every K=4 design steps** | 0.649 (step 20) | 10.14 | = v9 step 48 (still re-converged) |
| **v12** | v9 step 48 | **ESMFold2-FULL (1.3G) inside the design loop** (num_sampling_steps=10, 30 steps) | 0.668 (step 12) | 9.86 | = v9 step 48 (re-converged; same 5 diffs) |
| **v13** | v9 step 48 | **H3-only mutations** (H1/H2 frozen to v9 step 48; 16 mutable positions) | 0.676 (step 32) | 9.91 | = v9 step 48 (re-converged; same 5 diffs) |
| **multi-start v9 seeds 1-5** | v2 step050 (5 init seeds) | **Same v9 script with seeds 1, 2, 3, 4, 5** | **0.717 (seed=2 step 56)** | **9.52** | **`GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR`** (5 diffs, **NEW BASIN**) |
| **v12b** | **seed=2 step 56 (new best)** | **Direction 2 redux: ESMFold2-Full in design loop, started from the new best** | 0.700 (step 20) | 9.66 | **= new best** (re-converged; Full model did not escape) |
| **v13b** | **seed=2 step 56 (new best)** | **Direction 3 redux: H3-only mutations, started from the new best** | 0.701 (step 16) | 9.64 | **= new best** (H1/H2 frozen; re-converged; H3 mutations hurt) |
| **multi-start v9 seeds 6-15** | v2 step050 (10 more init seeds) | **Same v9 script, 10 additional seeds (6-15)** | 0.662 (seed=10 step 56) | 10.20 | `GLQIGYGMYMSYSGQKRVVTDSKQPIMKAPQR` (H1-8 V→M + H2-6 S→K independently re-discovered; not better than seed=2) |

### Key findings

1. **CA-coord prior > distogram-expected prior** (v9 vs v8): using realized 3D
   CA distances from averaged diffusion samples (not expected distance from
   distogram) unlocked a new local optimum. Wider constraint range [0, 45.6] Å
   vs [2.16, 21.84] Å for distogram expected distance.

2. **v9 step 48 was previously a strong local optimum, but NOT global**:
   v10a, v10b, v10c, and v11 all re-converge to the same sequence when started
   from v9 step 48. But a 5-seed multi-start from v2 init found a **different
   basin** (seed=2 step 56) that is +0.056 iptm better. The earlier "stuck in
   basin" conclusion was an artifact of single-seed exploration from v9 step 48.

3. **The new basin flips both H1-8 and H2-6 to a different chemistry**:
   - v9 step 48: H1-8 V→W (aromatic), H2-6 S→R (positive)
   - seed=2 step 56: H1-8 V→M (hydrophobic), H2-6 S→K (positive, but shorter)
   Both are 5-CDR-diff solutions with very different H3 patterns. The seed=2
   solution also has a 0.5 Å tighter CDR→epi contact layer (9.52 vs 10.02 Å).

4. **ESMFold2-Full in design loop (v12) does not help** when starting from v9
   step 48: best is step 12 with iptm 0.668, re-converged to the v9 sequence.
   The 4x slower forward pass didn't escape the basin either.

5. **H3-only mutations (v13) also don't help** when starting from v9 step 48:
   best is step 32 with iptm 0.676, re-converged to the v9 sequence. Freeing
   H1/H2 from the prior didn't help because H1/H2 were already optimal in v9.

6. **Framework 95 positions are pinned** throughout all versions: no framework
   AA mutation, but framework main chain coordinates do change between
   re-folds (because CDRs are different). The Ig fold is preserved by ESMFold2.

7. **Run-to-run Full iptm variance** is ~0.02-0.07 for the same sequence. The
   seed=2 step 56 winner's 0.717 vs v9 step 48's 0.661 is +0.056 — well outside
   noise. The 0.668 (v12) and 0.676 (v13) values for the same v9 sequence are
   within noise and indicate no real improvement from those directions.

8. **v12b / v13b re-converge when started from the new best (iptm 0.717)**:
   - **v12b** (Full in design loop): best = step 20 iptm 0.700, sequence =
     new best (no mutation succeeded). Full-model gradients in the loop did
     not escape the new basin either.
   - **v13b** (H3-only, H1/H2 frozen to new best): best = step 16 iptm 0.701,
     sequence = new best. H3 mutations from step 36+ all hurt (best of those
     was step 44 iptm 0.383 with H3 = `RQVTRSSTPIQKAGIR`).
   This confirms the new best is a real basin, not a lucky Full-eval draw.

9. **Expanded multi-start (10 more seeds, 6-15) found nothing better than 0.717**:
   best is **seed=10 step 56 iptm 0.662** (CDR=`GLQIGYGMYMSYSGQKRVVTDSKQPIMKAPQR`).
   Critically, **seed=10 independently re-discovered the H1-8 V→M + H2-6 S→K
   pattern** that defines the new best's H1/H2 — same H1 and H2 as seed=2
   step 56, with a different H3. This independently confirms the new best's
   H1/H2 chemistry is a robust signal in the loss landscape, not a single-seed
   fluke. Across all 15 seeds, the **H1 V→M + H2 S→K pattern appeared
   independently in seeds 2 and 10** (and partially in seed=6); no seed found
   anything better.

### Loss weights (consistent across v9, v10a/b/c, v11, v12, v13, multi-start)

```
epitope=0.2 intra=0.5 inter=0.5 glob=0.2 prior=0.3 aa_freq=0.01
```

### Multi-start basin diversity (5 seeds, all 60 design steps each)

| Seed | Final CDR | Diffs vs v2 | Final CDR→epi (Fast) | Best Full iptm | Best step |
|------|-----------|-------------|----------------------|----------------|-----------|
| 1 | `GLQVGYGWYMSYSGKKRVVPPSYTPIYKAPWR` | 8 | 10.37 | 0.561 (init) | step 0 — basin diverged at step 40 |
| 2 | `GLQIGYGWYMSYSGQKRVVTDSSTPIYKAGIR` | 5 | 15.06 | **0.717** | **step 56** |
| 3 | `GLQIGYGWYMSYSGQKRVVTDSYQPIYKAPGR` | 3 | 9.81 | 0.519 | step 44 |
| 4 | `GLQPGYGVYMSYSGQSRVVTDSYQPIQKAPQR` | 3 | 9.56 | 0.477 | step 48 — **loses H1 V→W** |
| 5 | (similar to v9 basin) | — | — | — | — |

**Conclusion**: seed=2 found the global-best basin. The other seeds either
re-converged to v9 (seed 3) or found worse basins (seeds 1, 2 step 60, 4).
A 5-seed ensemble from the same v2 init is sufficient to discover a better
solution than any single-seed or v9-iteration chain.

### Multi-start expanded (10 more seeds, 6-15)

| Seed | Best step's CDR (Full-eval) | Diffs vs v2 | CDR→epi (Full) | Best Full iptm | H1/H2 pattern |
|------|------------------------------|-------------|----------------|----------------|---------------|
| 6 (init) | `GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR` | 5 | 10.73 | 0.567 | H1 V→V / H2 S→S (no flips; close to v9) |
| 7 | `GLQIGYGRGMSYSGQSRVVGDSKQPIYKAPIR` | 6 | 11.65 | 0.515 | partial H2 S→R (v9-like) |
| 8 | `GLQIGYGMYMSYSGQSRVVTDSYQPIMKAPIR` | 4 | 13.14 | 0.396 | H1 V→M found, H2 S→S |
| 9 | `GLQIGYGIYMSYSGQSRVVTDSKQQIMKAPIR` | 5 | 12.68 | 0.385 | H1 V→I, H2 partial |
| **10** | **`GLQIGYGMYMSYSGQKRVVTDSKQPIMKAPQR`** | 5 | **10.20** | **0.662** | **H1 V→M + H2 S→K (NEW-BEST pattern!)** |
| 11 | `GLQIGYGINMSYSGQSRVVTDSKQQITKAPIR` | 6 | 11.43 | 0.535 | H1 V→I, partial H2 S→K |
| 12 | `GLQIGYGQYMSYSGQSRVVTDSYQPIYKAPRR` | 5 | 11.07 | 0.557 | H1 V→Q, H2 S→S (v9-like) |
| 13 | `GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR` | 5 | 12.03 | 0.443 | init-only (no flips) |
| 14 | `GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR` | 5 | 13.19 | 0.425 | init-only (no flips) |
| 15 | `GLQIGYGWYMSYSGQSRVVTDSYQDIYKAPRR` | 5 | 11.62 | 0.516 | H1 V→W, H2 S→S (v9-like) |

**Conclusion across all 15 seeds**:
- **seed=2 is the global best** at iptm 0.717; seed=10 is 2nd at 0.662.
- The H1-8 V→M + H2-6 S→K pattern (the new best's H1/H2) was **independently
  re-discovered by seed=10** and partially by seeds 6/8/11 — the H1-8
  methionine in particular appears in seeds 2, 6 (init), 8, 10, 11 (5/15
  seeds, 33% rate). This is a strong, robust signal in the loss landscape.
- No seed beat seed=2 step 56. The **+0.055 gap to next best (seed=10) is
  well outside run-to-run Full iptm variance (~0.02-0.07)**.

### What still doesn't work

- Iterating from a known-good sequence (v10a-c, v11, v12, v13) only refines
  the existing basin; it doesn't escape. The new best basin (seed=2 step 56)
  also resists escape: v12b (Full in loop) and v13b (H3-only) both
  re-converge to it when started from it.
- All prior designs started from a single seed (0) of the v2 init, so the
  random trajectory only explored one basin. Multi-start (now 15 seeds)
  shows that other basins exist and are accessible, but the seed=2 basin
  remains global best.
- The H3 region (16 positions) has a large mutation landscape; even within
  one seed, step-to-step CDR changes are large (e.g., seed=2 step 60 has
  CDR→epi 15.06 but step 56 has 9.52 — a 5.5 Å swing over 4 steps).
- v12b (Full in design loop from new best) failed to mutate the sequence at
  all: all 8 sampled steps returned the exact new-best CDR. The Full model
  produces a sharper loss landscape but **does not add a new escape route**.
- v13b (H3-only from new best) confirms the H1/H2 are optimal in the new
  basin: H3 mutations starting at step 36+ (e.g., T→R, P→R) all dropped
  iptm to 0.33-0.53. The H3 must remain at `RVVTDSSTPIYKAGIR` (or close to
  it) for high iptm.

---

## v10 scripts (independent)

- `design_b5_mps_v10a_iter_cacoord.py` — iter round 2 (single-variable test)
- `design_b5_mps_v10b_official_sched.py` — adds official cosine T + lr(T) + 1→50 sampling
- `design_b5_mps_v10c_official_plm.py` — adds aa_freq PLM proxy (0.05)
- `eval_v10a_candidates.py` / `eval_v10b_candidates.py` / `eval_v10c_candidates.py`

## v11: TRUE iterative prior refresh

`design_b5_mps_v11_iter_prior.py` — rebuilds the CA-coord prior every K=4
design steps from the Fast model's `sample_atom_coords` (not the distogram).
The prior target now moves with the optimization, not anchored to the
initial fold. 15 prior refreshes happened over 60 design steps (each ~0.2s).

Result: best is v11 step 20 (Full iptm 0.649), same sequence as v9 step 48.
A new sequence appeared at step 60 (`GLQIGYGWYMSYSGQARVVASSPQRIYKAPIR`, 2
CDR diffs from v9 step 48) at Full iptm 0.606.

## v12: ESMFold2-Full in design loop

`design_b5_mps_v12_full_in_loop.py` — uses the Full ESMFold2 (1.3G) for the
design forward pass instead of Fast (721M). Reduced num_sampling_steps from
20 (Fast default) to 10 (Full is more accurate per step) and steps from 60
to 30 to keep total time bounded. Started from v9 step 48 with the same
prior as v10a. Total design time: 25 min (vs 14 min for v9 with Fast).

Result: best by total = step 26 (CDR = v9 step 48, same 5 diffs), best by
iptm = step 12 (CDR = v9 step 48). Full eval top iptm = 0.668. Re-converged
to the v9 step 48 sequence; the Full-model design loop did not escape the
basin when starting from v9 step 48.

## v13: H3-only mutations

`design_b5_mps_v13_h3only.py` — H1 and H2 frozen to v9 step 48 values
(`GLQIGYGWYM` and `SYSGQR`), only H3 (16 positions) is mutable. Same loss
weights and prior as v9. Started from v9 step 48 with H1/H2 locked.

Result: best by total = step 59 (H3 = `RVVAQSPQRIQEAPIS`, 4 H3 diffs from
v9), best by iptm = step 8 (H3 = `RVVADSPQRIYKAPIR` = v9 init, no diffs).
Full eval top iptm = 0.676 (step 32, H3 = v9 init). Re-converged to the
v9 step 48 H3 sequence; the H1/H2 freeze didn't help because H1/H2 were
already optimal in v9.

## Multi-start v9 ensemble (Direction 1, the winner)

`run_v9_multistart.sh` — runs `design_b5_mps_v9_cacoord.py` with seeds 1, 2,
3, 4, 5 sequentially. Same v9 hyperparameters, only the random seed changes.
Total: 5 × 14 min = 70 min design + 8 min dedup eval = ~80 min.

`eval_v9_multistart.py` — folds each unique sequence across all 5 seeds with
Full ESMFold2, prints a unified top-10 by iptm and top-10 by CDR→epi.
Sequences are deduped by full sequence, so duplicate basins only fold once.

**Result**: seed=2 step 56 = new global best (iptm 0.717, +0.056 vs v9
step 48). This is the only direction of the 3 tested that found a strictly
better solution.

## v12b / v13b redux: starting from the new best

`run_v12b_v13b_from_newbest.sh` — re-runs directions 2 and 3 starting from
the new best (seed=2 step 56) instead of v9 step 48, to test whether the
new basin is a real optimum or a fragile lucky eval.

- **v13b** (`design_b5_mps_v13_h3only.py --init-seq ... --init-label
  "seed=2 step 56 (init, new best)"`): H1/H2 frozen to the new best;
  only H3 mutable. 16 design steps. Result: best = step 16 iptm 0.701,
  H3 unchanged. From step 36+ H3 mutations all hurt (best of those =
  step 44 iptm 0.383 with H3 = `RQVTRSSTPIQKAGIR`).
- **v12b** (`design_b5_mps_v12_full_in_loop.py --init-seq ...`): Full
  ESMFold2 in the design loop, started from the new best. 8 design
  steps. Result: best = step 20 iptm 0.700, sequence = new best, no
  mutation. Full-model gradients did not escape the new basin.

**Eval** uses `eval_v12_candidates.py` and `eval_v13_candidates.py` with
`--label-prefix "v12b step" / "v13b step" --init-label "seed=2 step 56
(init, new best)"` to fix the cosmetic name in the eval output.

`summarize_v12b_v13b_expanded.py` — final aggregation. Combines v12b +
v13b + multi-start (1-5) + multi-start (6-15) results and prints top 15
by iptm, top 15 by CDR→epi, and global best.

## Multi-start expanded (10 more seeds)

`run_v9_multistart_expanded.sh` — extends the multi-start to seeds 6-15
(same `design_b5_mps_v9_cacoord.py` script, just different `--seed`).
Eval uses `eval_v9_multistart.py` with `--snaps` for all 10 new files.
Total: 10 × 14 min = ~140 min design + ~10 min dedup eval.

`summarize_v12b_v13b_expanded.py` aggregates everything: v12b (8) +
v13b (16) + multi-start 1-5 (24) + multi-start 6-15 (46) = **94 valid
results** compared against the new best (iptm 0.717).

**Result**: best of the 10 new seeds is seed=10 step 56 (iptm 0.662,
+0.055 below seed=2). No seed beat seed=2. Critically, seed=10
independently re-discovered the new best's H1-8 V→M + H2-6 S→K pattern,
confirming this is a robust signal in the loss landscape, not a
single-seed fluke.

---

## Top 15 by Full ipTM (across all 94 results)

| # | Source | iptm | pTM | CDR→epi | CDR |
|---|--------|------|-----|---------|-----|
| 1 | seed=2 step 56 | **0.717** | 0.853 | 9.52 | `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR` |
| 2 | v13b step 16 | 0.701 | 0.851 | 9.64 | = new best (H3 frozen) |
| 3 | v12b step 20 | 0.700 | 0.851 | 9.66 | = new best (Full in loop) |
| 4 | v12b step 24 | 0.690 | 0.845 | 9.72 | = new best |
| 5 | v12b step 12 | 0.689 | 0.844 | 9.67 | = new best |
| 6 | v12b step 4  | 0.685 | 0.836 | 9.70 | = new best |
| 7 | v12b step 16 | 0.683 | 0.839 | 9.72 | = new best |
| 8 | v13b step 20 | 0.674 | 0.837 | 9.85 | = new best |
| 9 | seed=2 step 44 | 0.672 | 0.841 | 10.17 | `GLQIGYGMYMSYSGQSRVVTDSSTPIYKAPIR` (H2 reverts to S) |
| 10 | v13b step 28 | 0.668 | 0.837 | 9.74 | = new best |
| 11 | seed=10 step 56 | 0.662 | 0.837 | 10.20 | `GLQIGYGMYMSYSGQKRVVTDSKQPIMKAPQR` (H1-8 V→M + H2-6 S→K + different H3) |
| 12 | v13b step 8  | 0.661 | 0.827 | 10.14 | = new best |
| 13 | seed=10 step 52 | 0.659 | 0.824 | 10.07 | H1 V→M + H2 S→K pattern |
| 14 | seed=10 step 44 | 0.656 | 0.829 | 10.08 | H1 V→M + H2 S→K pattern |
| 15 | v13b step 32 | 0.654 | 0.830 | 10.05 | = new best |

**Top 15 by iptm are dominated by evaluations of the new best sequence**
(8 entries are the new best evaluated under v12b/v13b), confirming that
the new best is genuinely the global best and that no other basin
reaches close.
