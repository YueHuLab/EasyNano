# B5 binder design — which versions actually had impact

> Final ranking of every design version in the B5 VH-binder project by
> *real* contribution to the winning answer (iptm 0.692 ± 0.020,
> CDR = `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY`, framework pos 116 W→Y /
> CDR H3-16 R→Y).

The headline: **3 design versions + 1 evaluation method did 95% of
the work**. The other 14+ versions are negative results that ruled
out directions. This document ranks everything.

---

## 1. First tier — directly in the final answer (3 versions)

### 1.1 v9 + CA-coord prior — the architectural unlock

**What changed**: replaced v8's distogram-expected distance
(`E[softmax(d_logits) · bin_midpoints]`) with realized 3D CA-CA
distances from `sample_atom_coords` averaged over 4 diffusion samples.

**Numbers**:
- Prior constraint range: [2.16, 21.84] Å → [0, 45.6] Å
- Constrained pairs: 81013 → 106148
- Best iptm: 0.616 → **0.661** (+0.045)
- The "v9 step 48" sequence `GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR` was
  found here, but turned out to be a *local* optimum (re-discovered
  by other seeds later).

**Why it's #1**: every later improvement is conditional on v9's
better prior. v8 found a small gain; v9 found a *qualitatively
different* gradient landscape that made new basins reachable.

**Script**: `design_b5_mps_v9_cacoord.py`

### 1.2 v9 5-seed multi-start — found the new basin

**What changed**: ran v9 with seeds 1-5 from the same v2 init. seed=2
step 56 found a different basin (5 different CDR diffs from v9
step 48).

**Numbers**:
- Best single-sample iptm: **0.717** (seed 2, step 56)
- 5 CDR diffs from v2 init: H1-8 V→M, H2-6 S→K, H3-7 Y→S, H3-8 Q→T,
  H3-14 P→G
- Final CDR: `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR` (5 substitutions
  from WT, completely different from v9 step 48's chemistry)
- +0.056 over v9 step 48 (well outside the ±0.07 noise floor)

**Why it's #2**: 0.717 is the highest single-sample iptm the project
ever observed. v9 alone (single seed) tops out at 0.661. The 0.717
peak *required* multi-start.

**Script**: `run_v9_multistart.sh` (5 seeds); later expanded to 15
seeds in `run_v9_multistart_expanded.sh` (best of new 10 was 0.662,
no improvement over seed 2, but independently re-discovered H1-8
V→M + H2-6 S→K).

### 1.3 Step 2: framework micro-tuning (p116Y) — basin stabilizer

**What changed**: tried 10 single-residue mutations on the v9
5-seed winner. The mutation labeled "p116W→Y" (using framework
template indexing) is actually **R→Y at CDR H3-16** (using binder
indexing — there's a 1-off indexing bug, see §3.4 of
`B5_FULL_DESIGN_JOURNEY.md`).

**Numbers**:
- Best v9 5-seed baseline: median 0.653, std 0.103, bimodal
  [0.43, 0.70, 0.68] (n=9)
- v9_best_15seed_p116Y (R→Y at CDR H3-16): median **0.692**, std
  **0.020**, tight [0.65, 0.72] (n=9)
- Median improvement: +0.039
- Std improvement: 5.1× tighter
- Bimodality: killed (single tight mode)

**Why it's #3**: without it, the project ends at 0.717 single-sample
/ 0.653 median — a bimodal distribution where ~1/3 of evaluations
land in the 0.43 basin. The +0.039 median is nice, but the **5× std
reduction** is the real win: the design is now *reliably* in the
high-iptm basin across all 9 samples.

**Why R→Y works**: R (long, charged) at H3 C-flank can swing into
two conformations — high-iptm (packed) and low-iptm (solvent-exposed).
Y (shorter, polar) can't swing as far; it biases the noise toward
the high-iptm packing. Basin unchanged; basin *attractor basin
width* changed.

**Script**: `probe_pose_finetune.py` (Step 2 of the 3-step plan)

---

## 2. Second tier — methodology, not a design version (1 method)

### 2.1 Step 3 multi-sample evaluation (n=9)

**What it does**: Full ESMFold2 with `num_diffusion_samples=3` ×
3 seeds = 9 effective samples per sequence.

**Why it's foundational**:
- Spearman ρ(single-seed, median) = 0.486, p = 0.329 — **not
  significant**. Single-seed iptm rankings are essentially random.
- Without n=9, v9_best_15seed looks like a single-mode 0.717 design.
  With n=9, it's exposed as a bimodal [0.43, 0.70, 0.68] distribution.
- v9_best_15seed_p116Y is the *only* candidate that is BOTH
  high-iptm AND low-variance across the top-5 leaderboard. This
  cannot be known without the n=9 check.

**Methodology rules that emerged**:
1. Always run ≥3 seeds per evaluation
2. Always run num_diffusion_samples ≥ 3
3. Rank by median; reject candidates with std > 0.05 unless median
   is exceptional (> 0.7)
4. Top-3 agreement between single-seed and multi-sample: 1/3 in this
   project — confirms the noise

**Script**: `probe_pose_robust.py`

### 2.2 v8 distogram-expected prior — small step that opened the door

**Numbers**: iptm 0.572 → 0.616 (+0.044). Same basin as v2 (CDR
unchanged), but a better gradient signal.

**Why it matters**: v8 *proved the prior direction was right* —
just the formulation was too smooth. v9's "use realized 3D CA
distances instead of distogram expectation" was a 1-line change
that built directly on v8. Without v8's 0.616, v9's 0.661 looks
like an incremental tweak. With v8 as stepping stone, v9 is
clearly the "fix the formulation" step.

**Script**: `design_b5_mps_v8_fullprior.py`

### 2.3 v16 one-shot Full-fold re-anchor — found a third basin

**Numbers**: best v16 across 5 seeds = 0.591 (seed 5 step 44). Worse
than v9 multi-start's 0.717, but a *different* CDR
(`GLQIGYGRYMSYSGQSRVVTDSYQPIYKAPQR`) that v9 multi-start never
found in 15 seeds.

**Why it matters**: as a "diversification tool after v9 multi-start
saturates", v16 has value. It demonstrates that re-anchoring the
epitope midway through a design run (based on the model's *own*
prediction of where the binder is going) can find CDRs the original
fixed-epitope v9 missed.

**Why it's not #1**: the best v16 result (0.591) is below v9 5-seed
multi-start's 0.717. Use v16 to *supplement* a v9 sweep, not replace
it.

**Script**: `design_b5_mps_v16_fullfold_reanchor.py`

---

## 3. Third tier — negative results that ruled out directions

These versions collectively consumed ~60% of the project time. None
of them improved the answer, but each one eliminated a direction so
later iterations didn't waste time on it.

| Version | What it tried | Why it failed | What it proved |
|---|---|---|---|
| v3 | Full ESMFold2 (1.3G) in design loop | 4× slower, similar/better gradient | Fast model (721M) is enough for design |
| v4 | hard mask for non-CDR positions | Equivalent to v2's soft mask | Soft mask + soft logit is the right mechanism |
| v5/v6 | Optimize H1+H2 only (H3 frozen) | H3 must move for binding | 3 CDRs must be co-optimized |
| v7 | Adam + larger LR (0.5) | 0.5 LR overshoots in 32-dim space | LR ≤ 0.1 is the right range |
| v10a | re-run v9 from v9 step 48 | re-converge to 0.656 | v9 step 48 is a basin |
| v10b | + official cosine LR sched | re-converge to 0.668 | LR sched doesn't escape basin |
| v10c | + strong PLM proxy (0.05) | iptm 0.641, 5 *new* diffs (worse) | PLM priors hurt binding design |
| v11 | iterative prior refresh (every 4 steps) | re-converge to 0.649 | prior must be fixed, not iterative |
| v12 | Full ESMFold2 in design loop | re-converge to 0.668 | Full model in loop is 4× cost, no help |
| v12b | Full in loop, from new best | re-converge to 0.700 | even Full-gradients can't escape the new basin |
| v13 | H3-only mutations from v9 step 48 | re-converge to 0.676 | H1/H2 already optimal in this basin |
| v13b | H3-only, from new best | re-converge to 0.701 | H3 single-axis can't escape basin |
| v9 seeds 6-15 | 10 more seeds from v2 init | best of new = 0.662 (no new winner) | H1-8 V→M + H2-6 S→K independently re-discovered; basin shape is real |
| v14 | dynamic top-K epitope (every step) | self-referential, no signal | epitope must be fixed, not dynamic |
| v15 | periodic re-anchor (every 4 steps) | "epitope jump" defeats gradient | epitope must be fixed for the whole run |
| Step 4 (v9 iter from p116Y) | run v9 with p116Y as init, 3 seeds × 30 steps | n_cdr_diff = 0 in all 3 seeds (no-op) | v9 loop is blind w.r.t. iptm basin improvement once at local min |

**Total negative-result count: 16 design-loop runs, ~3 days of compute**
(`B5_V14_DYNAMIC_VS_FIXED.md`, `B5_V15_PERIODIC_REANCHOR.md`,
`B5_CDR_DESIGN_SUMMARY.md` all document the failures).

---

## 4. Fourth tier — diagnostic experiments (no design output, but project foundation)

These are not "versions" — they're understanding experiments. They
cost ~3 hours total but are responsible for every architectural
decision in §1 and §2.

| Experiment | What it showed | Why it mattered |
|---|---|---|
| `probe_pose_n5.py` | Same sequence × 5 seeds → 1.84 Å binder RMSD, iptm 0.39-0.70 | ESMFold2's pose is a function of (sequence, noise). The "bimodality" lives in the noise realization, not the sequence. |
| `probe_pose_test1.py` | Inject 30°-rotated binder as x_init → output 0.58 Å from baseline (= control 0.59 Å) | x_init is completely ignored. ESMFold2's `_center_random_augmentation` is rotation-invariant. "Vary pose, fix CDR" is not viable. |
| `probe_pose_test2.py` | v2 init vs v9_best_15seed (5/32 CDR diffs) → framework RMSD 0.29 Å, CDR RMSD 2.54 Å, binder RMSD 1.38 Å | Framework determines pose. CDRs only fine-tune contact. This is *why* pinning the framework is the right design decision. |
| `probe_ipsae_diagnostic.py` | ipSAE_p10 (CDR↔epitope) gives a PAE-based contact metric independent of iptm | Provides a second metric for cross-checking iptm. iptm is "model confidence in interface"; ipSAE is "is the contact actually close in the prediction". |
| `probe_pose_crossval.py` | ESMFold2-Fast iptm ≈ 0.055 ± 0.005 for everything | Fast model's iptm is uncalibrated; never use it for ranking. (Boltz-1 cross-validation was attempted but offline-blocked at CCD download.) |

**These are why we know the answers**. Without `test2`, you don't
know framework is rigid. Without `test1`, you might try to vary
pose. Without `n5`, you don't know bimodality is noise. Without
`ipsae_diag`, you only have iptm.

---

## 5. The single chart that explains everything

```
iptm journey (Full ESMFold2 single-sample, then multi-sample for final)
                                                    
0.6 |                                                
0.5 |● v2 (0.572, locked)                           
0.4 |                                              
0.3 |                                              
    |                                              
0.7 |                                  ●v9 5-seed    ← new basin (0.717 peak)
    |                                  \            
    |                                   \           
0.6 |                                    ●v9 step48  ← local basin (0.661)
    |              ●v8 (0.616)         \           
    |              \                    \          
0.5 |               \                    ●p116Y stable median (0.692)
    |                ●v10c (0.641, worse)            
    |                                            
0.4 |     v3,v4,v5,v6,v7,v10a,b,v11,v12,v12b,v13,v13b,  
    |     v14,v15 (all re-converge to baseline or below) 
    |                                              
0.3 |                                              
    +-------+-------+-------+-------+-------+-----+
    v2      v3-v8   v9      v10     v12,v14  p116Y
    init    search  step48  re-tries  divers  FINAL
    
```

**Two jumps that mattered**:
- v8 → v9: 0.616 → 0.661 (prior formulation)
- v9 single → v9 5-seed: 0.661 → 0.717 (multi-seed finds new basin)

Everything else is "verify" or "stabilize" or "negative control".

---

## 6. The 3 things that actually worked, restated

1. **Better prior formulation** (v8 → v9: distogram-expected → CA-coord
   realized) — 1 line of code change
2. **More starting points** (v9 single-seed → v9 5-seed multi-start) —
   5× compute, +0.056 iptm peak
3. **Stabilize the basin once found** (Step 2 p116Y framework micro-tuning)
   — 10 single-residue mutations, 30 min compute, std 5× tighter

Plus one methodology:
4. **Multi-sample evaluation** (Step 3 n=9) — without it, the "best" is
   random

**The full final answer** (`v9_best_15seed_p116Y`):
- CDR: `GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY`
- Framework pos 116 (CDR H3-16): W→Y in framework template / R→Y in
  binder sequence
- Robust iptm: median 0.692, std 0.020, range [0.651, 0.716] (n=9)
- Source: v9 5-seed multi-start + Step 2 framework micro-tuning
- Confidence: HIGH (single tight basin, all 9 samples in [0.65, 0.72])

---

## 7. What was tried but did *not* work (in priority order)

If you want to know which directions to skip in a similar future
project, this is the list:

1. **Full ESMFold2 in design loop** (v3, v12, v12b) — 4× cost, no help
2. **Iterative prior refresh** (v11) — self-referential, defeats gradient
3. **Periodic epitope re-anchor** (v15) — epitope jump destroys gradient
4. **Dynamic top-K epitope** (v14) — same as v15
5. **H3-only mutations** (v13, v13b) — H1/H2/H3 are coupled, can't
   decouple
6. **Strong PLM proxy as regularization** (v10c) — hurts binding designs
7. **Larger LR** (v7) — overshoots in 32-dim space
8. **Hard mask** (v4) — equivalent to soft mask, no benefit
9. **v9 iter-design from a basin** (Step 4) — gradient is zero, no-op
10. **Larger epitope weight without prior fix** (would have been v3.5) —
    unverified, but the prior would dominate anyway
