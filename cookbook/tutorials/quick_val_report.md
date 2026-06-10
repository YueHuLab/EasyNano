# Quick-Validation Report: v2 design loop on 5 common targets (3 + 2 alt framework)

> **Date:** 2026-06-08
> **Question:** Does the v2 epitope-attraction design loop generalize from B5 to other common targets?
> **Panel:** PD-L1 (5JDS) / RBD (6WAQ) / TNFα (5M2J) / RBD (6ZXN, alt) / TNFα (5M2M, alt), 5 seeds × 100 steps each. The first 3 use the target's own WT VHH framework; the last 2 use an alternative VHH framework (Ty1 for RBD, VHH3 for TNF) to test cross-framework generalization.

---

## Setup

For each target, we used:
- The WT VHH **from the target's own crystal structure** (KN035 / VHH-72 / anti-TNF VHH).
- The corresponding **VHH-contact epitope** (target residues within 8 Å of the WT VHH heavy atoms).
- **5 seeds × 100 steps** of v2 design.
- Fast ESMFold2 (721M) for the design loop, **Full ESMFold2 (1.3G)** for evaluation.

This is a **self-recovery test**: the framework's pose is correct for the target
(per [[esmfold2_pose_basin]]), and v2 should improve on the WT CDRs.

### Notable parameter choices

- `--allow-cdr-cys` for PD-L1 (KN035 H3 contains a structural disulfide; default would pin Cys in CDR and break it).
- All other parameters identical to B5 v2 defaults (lr=0.5, T cosine schedule, w_epitope=0.05).

### Design loop metric definitions

- `L_epi` = mean over CDR rows of `ELU(min_dist_to_epitope - 8.0)`. Drops as CDRs approach the epitope.
- `CDR→epi (min)` = average over CDR rows of the minimum predicted distance to any epitope residue (Å). Lower is closer.
- `inter` = binder-target inter-contact entropy loss.
- `grad norm` = L2 norm of the gradient on the soft logits (peaks early, decays as we hit a basin).

---

## Results

### Per-target design-loop summary

(Each cell is the best snapshot from each seed, picked by `inter` loss; values reported as median [min, max].)

| Target | Framework | L_epi step 0 | L_epi step 100 (best) | CDR→epi step 0 (Å) | CDR→epi step 100 (Å) | n_mutated | #steps to basin |
|---|---|---|---|---|---|---|---|
| PD-L1 (5JDS) | KN035 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6WAQ) | VHH-72 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2J) | anti-TNF | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6ZXN) | Ty1 (alt) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2M) | VHH3 (alt) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Full ESMFold2 evaluation (5 seeds × 11 snapshots × 5 targets)

| Target | WT init iptm | WT init pTM | Best design iptm (median of 5-seed peak) | Best design pTM | iptm Δ (design − WT) | n-folded |
|---|---|---|---|---|---|---|
| PD-L1 (5JDS) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6WAQ) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2J) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6ZXN) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2M) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Cross-validation against known VHH crystal structure (Kabsch on target CA)

| Target | init (WT) binder-RMSD vs real VHH (Å) | design binder-RMSD vs real VHH (Å) | Δ (des−init) (Å) | target-RMSD (sanity, Å) | n_interface contacts (pred / real) |
|---|---|---|---|---|---|
| PD-L1 (5JDS) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6WAQ) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2J) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RBD (6ZXN) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TNFα (5M2M) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

RMSD computed as: predicted binder CA vs real VHH CA, after Kabsch alignment of predicted target CA to real target CA (in the source PDB).

**Note on interpretation**: per [[esmfold2_pose_basin]], the *absolute* binder-RMSD vs crystal can be very large (10s of Å) even for the WT, because ESMFold2's predicted pose for a given framework can differ from the crystal pose. The relevant metric is the **Δ (des−init)**: a small delta means the design preserved the framework's predicted pose. Large deltas (≥ ~5 Å) would indicate pose drift.

---

## Pre-run smoke test (single-seed 30-step, PD-L1 / KN035)

This motivated the full panel — design improved iptm over WT on the smoke test:

| Source | iptm | pTM | CDR→epi (Å) | CDR seq |
|---|---|---|---|---|
| KN035 WT init | 0.221 | 0.573 | 14.13 | `GKMSSRRLTTSGSDSFEDPTCTLVTSSGAFQY` |
| seed 0 / step 10 | 0.326 | 0.616 | 12.65 | `GKQSSRRNWVDGPFLVELPDCELVSFIGFYPY` |
| seed 0 / step 30 | 0.297 | 0.619 | 12.86 | `GQTSARQNWAPGPFELELPDDELVSFEGFYFY` |
| B5 framework on PD-L1 (control) | 0.172 | 0.540 | 15.39 | B5 init CDRs (no design) |

The B5-framework control is a **framework-pose negative control**: B5's framework has the wrong pose for PD-L1, and the iptm is lower than KN035's WT (0.172 vs 0.221). This confirms [[esmfold2_pose_basin]]: framework determines pose, and the framework must be in a "right" pose for the design to work.

---

## Findings (to be filled in after full panel completes)

1. **Does v2 work for arbitrary new targets?** _TBD_
2. **Are the improvements real or noise?** _TBD_
3. **Is the pose preserved (binder-RMSD ≤ ~5 Å to real VHH)?** _TBD_
4. **What are the failure modes, if any?** _TBD_

## Caveats and known limitations

1. **Framework pose is critical**: per [[esmfold2_pose_basin]], the framework determines the predicted pose. v2 cannot easily move CDRs to an epitope that's not reachable from the framework's natural pose. A "wrong" framework gives a low iptm baseline.
2. **Cys-pinning in CDRs**: the default `--pin-cys-in-cdr` (B5 convention) prevents the design from putting Cys in CDR positions. For VHH frameworks with H3 disulfides (e.g., KN035), use `--allow-cdr-cys`.
3. **5 seeds may not be enough**: the B5 multi-start work used 15 seeds and found basins ranging from 0.43 to 0.72. Our 5 seeds may miss the global minimum.
4. **No initial binder pose**: for new targets we don't pre-dock the framework. The design starts with the framework's default pose (whatever ESMFold2's prior predicts for that framework alone). For some targets this may be very far from the target.

## Files

- `/tmp/quick_val/runs/<TAG>_seed<N>.json` — design-loop snapshots
- `/tmp/quick_val/evals/<TAG>_seed<N>_eval.json` — Full ESMFold2 evaluations
- `/tmp/quick_val/evals/_all_targets_summary.json` — per-target step-level aggregates
- `/tmp/quick_val/rmsd_summary.json` — Kabsch cross-validation
- `cookbook/tutorials/test_target_pdb.py` — generalized setup
- `cookbook/tutorials/design_target.py` — generalized design loop
- `cookbook/tutorials/eval_target_snapshots.py` — generalized eval
- `cookbook/tutorials/eval_quick_val.py` — batch eval across panel
- `cookbook/tutorials/rmsd_to_native.py` — Kabsch cross-validation
- `cookbook/tutorials/run_quick_val.sh` — panel runner
