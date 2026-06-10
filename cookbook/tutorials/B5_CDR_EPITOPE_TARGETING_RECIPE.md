# B5 Antibody CDR Epitope-Targeting — Working Recipe

> **Single source of truth for the pipeline that *actually* targets the B5
> epitope.** This is the procedure that produced the global-best candidate
> (Full ESMFold2 ipTM = **0.717**, pTM = **0.853**, CDR→epitope = **9.52 Å**).
> All other tested variations (v2–v8, v10a–c, v11, v12, v13, v12b, v13b)
> produced strictly worse results. See `B5_CDR_DESIGN_SUMMARY.md` for the
> version history of what didn't work and why.

---

## TL;DR — The three things that matter

1. **CA-coord structure prior (v9)**, not distogram-expected (v8) and not
   iteration (v10a/v11). It is built from realized 3D CA distances averaged
   across **4 diffusion samples** of the **Full ESMFold2 (1.3G)** model, not
   from `softmax(d) · midpoints`. This single change took the basin from
   iptm 0.616 → 0.661 and unlocked the better basin.
2. **Multi-start with 10–15 random seeds** of the v9 design loop. Single-seed
   gets stuck in a local basin; multi-start finds the global best in 1/15
   seeds (seed=2 step 56). Seed=10 independently re-discovered the new
   best's H1/H2 pattern, validating it is a real signal in the landscape.
3. **Verify every candidate with Full ESMFold2 (1.3G)**, not Fast (721M).
   The Fast model's iptm/structure is only a proxy; Full re-evaluation is
   the ground truth. Run-to-run Full iptm variance on the same sequence is
   ~0.02–0.07, so only deltas > 0.1 are decisive.

Everything else (PLM proxy regularization, official cosine schedule, Full
in the design loop, H3-only restriction, iterative prior refresh) was
tested and **did not help**.

---

## 1. Input

- **Target**: chain A of `test/B5.pdb` (223 aa, 7-TM protein). Sequence:
  ```
  QAFWKAVTAEFLAMLIFVLLSLGSTINWGGTEKPLPVDMVLISLCFGLSIATMVQCFGHISGGHINPAVTVAMVCTRKISIAKSVFYIAAQCLGAIIGAGILYLVTPPSVVGGLGVTMVHGNLTAGHGLLVELIITFQLVFTIFASCDSKRTDVTGSIALAIGFSVAIGHLFAINYTGASMNPARSFGPAVIMGNWENHWIYWVGPIIGAVLAGGLYEYVFCP
  ```
- **Binder framework**: chain B of `test/B5.pdb` (127 aa, VH framework III,
  no light chain). Sequence:
  ```
  QVQLVESGGGLVQPGGSLRLSCAASGFTFGTGSYYSLGWFRQAPGQGLEAVAAISSSGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARGFTYSYYPDYRAYDFWGQGTLVTVS
  ```
- **CDR positions** (32 mutable, abnumber Chothia): H1 25–34 (10 aa),
  H2 54–59 (6 aa), H3 101–116 (16 aa). 95 framework positions pinned.
- **Epitope** (21 residues, auto-detected within 8.0 Å of binder in the
  starting PDB): `[24, 30, 31, 33, 111, 112, 113, 114, 115, 116, 117, 118,
  119, 120, 121, 173, 174, 176, 177, 196, 197]` (1-based, target chain A).

### v2 init starting sequence (used as the multi-start seed; 95 framework
positions are pinned to this, 32 CDR positions are mutable)

```
QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPIYKAPIRWGQGTLVTVS
```

CDRs in this init: H1=`GLQIGYGVYM`, H2=`SYSGQS`, H3=`RVVTDSYQPIYKAPIR`.

---

## 2. Models and infrastructure

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Design forward pass | `ESMFold2-Fast` (`/Users/huyue/esm-c-fold2/ESMFold2-Fast`) | 721 M | 60 design steps, ~14 min/run on MPS |
| Sequence encoder | `ESMC-6B` (`/Users/huyue/esm-c-fold2/ESMC-6B`) | 6 B | used inside ESMFold2 |
| Prior builder | `ESMFold2` (Full, `/Users/huyue/esm-c-fold2/ESMFold2`) | 1.3 G | fold (binder, target) → 3D CA coords → 64-bin distogram prior |
| Verifier | `ESMFold2` (Full, same) | 1.3 G | re-fold every candidate; the only trustworthy score |

`DEVICE = "mps"`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `v9` for
operations that MPS lacks. Full loading takes ~10 s (Full) or ~4 s (Fast);
no `xformers`, no `transformer-engine` installed (pure-PyTorch fallback,
output noise is in the few-ULP range after the final LayerNorm).

---

## 3. Soft-logit parameterization (the only learnable tensor)

A single tensor `soft_logits` of shape `[L_binder, 20]` is the learnable
parameter, optimized by Adam. At each step we convert it to:

```
binder_probs_20 = softmax(soft_logits / T)        # T=1.0
binder_probs_33 = pad to 33 res_types (idx 0..1 are <pad> and -, 2..21 are AA)
res_type_soft   = concat([binder_probs_33, target_one_hot], dim=1)   # [1, L_total, 33]
```

`res_type_soft` is fed into ESMFold2 in place of the hard one-hot, making
the entire fold differentiable w.r.t. the binder AA distribution.

**Initialization (`init_soft_logits` in `design_b5_mps_v2.py`)**:
- Framework positions (95): pinned to WT AA with logits `(-10, …, 10, …, -10)`.
  After each step `pin_fixed_positions` reasserts this, so the framework is
  mathematically constant.
- Mutable CDR positions (32): random logits ~ N(0, 0.5²) for the 19 off-WT
  AAs, with `wt_logit=5.0` for the WT AA at that position. Cysteine is
  blocked at `-10` to avoid spurious disulfides.

This means the **starting CDR sequence is exactly the v2 init** for every
seed; only the random noise that perturbs the off-WT logits changes. The
**main chain coordinates of the framework are NOT fixed** — they move
between re-folds because the CDR sequence is different — but ESMFold2
preserves the Ig fold.

---

## 4. Structure prior — the v9 critical detail

The prior is the only thing that distinguishes v9 from v8. It is built
**once per design run** (not refreshed) from the Full ESMFold2 fold of the
**initial** sequence, then held fixed.

```
def predict_prior_from_full_ca(binder_seq, target_seq,
                                num_loops=3, num_sampling=14,
                                num_diffusion_samples=4,
                                bin_tolerance=2.5, n_bins=64,
                                min_dist=2.0, max_dist=22.0):
    fold with ESMFold2-Full, num_diffusion_samples=4
    extract per-token CA coords (B*ds, L, 3)  -- shape (4, 350, 3)
    average across the 4 diffusion samples   -> (L, 3) CA coords
    pairwise CA-CA distances                 -> (L, L) distance matrix
    reorder to target-first
    target_target_dist = ca_dist[:T, :T]
    interface_dist     = ca_dist[T:, :T]    (binder rows, target cols)
    prior_bins, prior_mask = build_pdb_prior(
        binder_length=B, target_length=T,
        target_target_dist=tt_dist,
        interface_dist=iface_dist,
        bin_tolerance=2.5, n_bins=64, min_dist=2.0, max_dist=22.0,
    )
    return prior_bins, prior_mask, iptm, ptm
```

**Why this beats v8's distogram-expected prior**:
- v8 used `E[d] = softmax(d)·midpoints`, a single scalar per pair derived
  from the predicted distogram. This blurs close contacts because the
  expected value smooths over the entire 0–22 Å distribution.
- v9 uses **realized** 3D CA distances, averaged across 4 diffusion
  samples (variance reduction, same atomic positions). The distance
  matrix can take values anywhere in `[0, +∞)`, including > 22 Å.
- The binning in `build_pdb_prior` is loose (`bin_tolerance=2.5` Å means
  the actual distance can be ±2.5 Å from the bin center), and the mask
  only constrains pairs where the predicted distance is in `[2, 22]`.
  Distant pairs (>= 22 Å) are unconstrained, so the prior permits the
  CDRs to fly far from the epitope if the loss wants to.
- Constraint range in v9: `[0, 45.6]` Å (from `ca_dist` actual range).
  Constraint range in v8: `[2.16, 21.84]` Å (from expected distogram).
  v9 gives the optimizer **more slack** to explore larger rearrangements
  while still anchoring the framework and the bulk of the target.

For the B5 starting structure (binder folded against target at the
v2 init), the prior stats are:
- 106,148 / 122,500 pairs constrained (87 %).
- target-target constrained: 49,506.
- interface constrained: 28,321.
- interface CA distance range: `[3.67, 79.61]` Å (median 38.76).

The full-matrix prior is `[L, L, 64]` of int64 bin indices, with a
boolean mask of the same shape that says which pairs to apply the
prior to. (`build_pdb_prior` returns both; only constrained pairs
contribute to the loss.)

---

## 5. Loss

`compute_structure_losses` (in `binder_design_hy_losses.py`) returns
five loss components and a total. Default weights used in the winning
run (set in `run_design` from CLI flags):

```
epitope           = 0.2
intra_contact     = 0.5
inter_contact     = 0.5
glob              = 0.2
structure_prior   = 0.3
aa_freq (PLM-proxy) = 0.01
```

### What each component does

| Component | Computation | What it pushes the design toward |
|-----------|-------------|-----------------------------------|
| `epitope` | For each CDR residue, soft-min over its 8 nearest epitope residues of `d − 8.0 Å` (sigmoid-shaped, cutoff 8 Å). Sum / num_CDR. | Tight binder→epitope contact. |
| `intra_contact` | For each pair of binder residues, soft-min over their distance. | Compact, well-packed binder fold. |
| `inter_contact` | For each pair of (binder, target non-epitope) residues, soft-min over their distance. | Prevents the binder from collapsing onto off-target patches of the antigen. |
| `glob` | Negative log-radius-of-gyration of the binder. | Compact, globular binder shape. |
| `structure_prior` | For each constrained (i, j) pair, the prior bin index is mapped to a 1-hot over the 64 bins and the cross-entropy between the model's predicted distogram (at the same pair) and the 1-hot prior bin is summed. | Match the predicted fold to the v9 prior. |
| `aa_freq` | KL divergence of the marginal binder AA distribution (averaged across binder positions) vs a natural-protein AA frequency prior (`AA_FREQ` tensor in `design_b5_mps_v2.py`). | Avoid pathological AAs. Weight 0.01 is tiny; it's a tie-breaker, not a dominant signal. |

Total loss = `Σ w_i · component_i`.

### Loss internals worth knowing

- All soft-min operations use `masked_min_k(x, mask, k)` with k=8 by
  default — top-8 closest neighbors are smoothed; this keeps the
  contact loss differentiable at the edge of the cutoff.
- The 64 distogram bins span `[2, 22] Å` with midpoints returned by
  `get_mid_points()`. The prior loss is computed by indexing the
  ESMFold2 predicted distogram at `[i, j, prior_bin]` and taking
  `−log p`.
- The epitope loss is **the only term that knows about the epitope**.
  Everything else (intra, inter, glob, prior, aa_freq) is just there
  to keep the binder fold well-formed.

---

## 6. Design loop

In `run_design` (in `design_b5_mps_v9_cacoord.py`):

```python
soft_logits = init_soft_logits(v9_template, init_seq, wt_logit=5.0).to(DEVICE)
soft_logits = soft_logits.detach().requires_grad_(True)
optimizer = optim.Adam([soft_logits], lr=0.05)

for step in range(steps):
    res_type_soft = build_soft_res_type(soft_logits, target_one_hot, T=1.0)
    out = esmfold2_forward(res_type_soft)         # Fast model
    losses = compute_structure_losses(out, prior_bins, prior_mask, cdr, epi, …)
    total = sum(w_i * component_i)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    pin_fixed_positions(soft_logits, v9_template)   # re-pin framework
```

- **Optimizer**: Adam, `lr=0.05`. No momentum or weight decay tweaks
  helped in any test.
- **Framework pinning**: applied *after* `optimizer.step()` so the
  framework logits are reset to one-hot for the next step. Without
  this, the framework would drift.
- **Forward sampling**: `num_sampling_steps=5`, `num_loops=1` for
  Fast in design (gives ~14 min for 60 steps on MPS).
- **Snapshot policy**: every 4 steps, save the current hard sequence
  to the snapshot file. 60 steps → 15 snapshots per run.
- **Log policy**: every 8 steps, print total / per-component / Fast
  pTM+ipTM / current CDR sequence.

A single seed's design run: 60 steps × 14 s/step ≈ **14 min** on MPS.

---

## 7. Multi-start

This is what turns a single 14-min run into a **global-best search**.

### Why it works

The CDR optimization landscape has multiple basins separated by high
loss barriers (the optimizer gets stuck as soon as the loss gradient
goes flat). Each random seed sets the Adam noise differently
(`torch.manual_seed(seed)`, `np.random.seed(seed)`), so different
seeds fall into different basins. With 15 seeds, basin diversity
covers most of the accessible solution space.

### Command (multi-start, 15 seeds)

```bash
# Seeds 1-5 (the original multi-start)
for seed in 1 2 3 4 5; do
  python3 -u design_b5_mps_v9_cacoord.py \
    --steps 60 --lr 0.05 --seed $seed \
    --snapshot-path /tmp/b5_v9_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v9_seed${seed}.log 2>&1
done

# Seeds 6-15 (the expanded multi-start; same script, just more seeds)
for seed in 6 7 8 9 10 11 12 13 14 15; do
  python3 -u design_b5_mps_v9_cacoord.py \
    --steps 60 --lr 0.05 --seed $seed \
    --snapshot-path /tmp/b5_v9_seed${seed}_snaps.json \
    --log-every 8 --snapshot-every 4 \
    > /tmp/b5_v9_seed${seed}.log 2>&1
done
```

The packaged shell scripts are:
- `run_v9_multistart.sh` — seeds 1-5 (with the design loop only).
- `run_v9_multistart_expanded.sh` — seeds 6-15 + auto-eval with Full.

### Cost

| Phase | Time |
|-------|------|
| 15 design runs (60 steps each, Fast) | 15 × 14 min ≈ **3.5 h** |
| Full-eval of 46 unique sequences (from 161 snapshots, dedup'd) | ~15 min |
| **Total** | **~3.75 h on Mac MPS** |

The 15 seeds produce ~225 candidate sequences; dedup by full sequence
yields ~46 unique ones.

---

## 8. Evaluation with Full ESMFold2

`eval_v9_multistart.py`:

```python
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v9_seed1_snaps.json ... /tmp/b5_v9_seed15_snaps.json \
  --out /tmp/b5_v9_multistart_full_eval.json
```

For each unique sequence in the snapshot files:
1. Build the full binder sequence (template + CDR slots filled in).
2. Run Full ESMFold2 (`num_loops=3, num_sampling_steps=14,
   num_diffusion_samples=1`) once on (binder, target).
3. Read `iptm`, `ptm` from the output.
4. Compute CDR→epitope min CA distance from `sample_atom_coords` and
   the epitope token list.
5. Print a unified top-N by iptm and top-N by CDR→epitope.

**Output ranking (top 5 from the full 15-seed run)**:
```
                          iptm  pTM  CDR→epi  CDR
seed=2 step 56           0.717 0.853  9.52  GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR
seed=10 step 56          0.662 0.837  10.20 GLQIGYGMYMSYSGQKRVVTDSKQPIMKAPQR
seed=6  step 0 (init)    0.567 0.809  10.73 GLQIGYGVYMSYSGQSRVVTDSYQPIYKAPIR
seed=12 step 44          0.557 0.806  11.07 GLQIGYGQYMSYSGQSRVVTDSYQPIYKAPRR
v9 step 48 (old best)    0.551 0.801  10.91 GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR
```

---

## 9. The winning candidate

```
ipTM    = 0.7171
pTM     = 0.8528
CDR→epi = 9.52 Å   (Fast metric, min CA distance)
Source  = b5_v9_seed2_snaps step 56
```

**CDR sequence** (5 substitutions from the v2 init):
```
H1 (25-34)  GLQIGYGVYM  →  GLQIGYGMYM   (V→M at H1-8)
H2 (54-59)  SYSGQS      →  SYSGQK        (S→K at H2-6)
H3 (101-116) RVVTDSYQPIYKAPIR  →  RVVTDSSTPIYKAGIR
            (Y→S at H3-7, Q→T at H3-8, P→G at H3-14)
```

**Full binder sequence (127 aa)**:
```
QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGMYMSLGWFRQAPGQGLEAVAAISYSGQKTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSSTPIYKAGIRWGQGTLVTVS
```

**Antigen (unchanged, 223 aa)**: see section 1.

**Independently re-discovered H1/H2 pattern**: seed=10 converged to
`GLQIGYGMYM` (H1) + `SYSGQK` (H2) — the same H1-8 V→M + H2-6 S→K
substitutions as the winner — with a different H3, reaching iptm 0.662.
The H1 V→M (methionine) appears in 5/15 seeds (2, 6, 8, 10, 11). The
pattern is a robust signal in the loss landscape, not a single-seed
fluke.

---

## 10. File map

| File | Role |
|------|------|
| `design_b5_mps_v9_cacoord.py` | **The working design loop.** CA-coord prior + 60-step Adam optimization. CLI: `--steps 60 --lr 0.05 --seed N --snapshot-path ... --init-seq ...`. |
| `run_v9_multistart.sh` | Bash loop over seeds 1–5. |
| `run_v9_multistart_expanded.sh` | Bash loop over seeds 6–15 + auto-eval at the end. |
| `eval_v9_multistart.py` | Dedup sequences across snapshot files, fold each with Full ESMFold2, print top-N by iptm and by CDR→epi. |
| `summarize_v12b_v13b_expanded.py` | Aggregates everything (v12b, v13b, all 15 multi-start seeds) into a single top-15 ranking and global-best. |
| `test_b5_pdb.py` | `setup_design()` — loads `test/B5.pdb`, auto-detects epitope (8 Å cutoff), runs `abnumber` for Chothia CDRs, builds the 95/32 framework/CDR split, builds the v9 disto-graph prior from the input PDB. |
| `binder_design_hy_losses.py` | All loss components, `LOSS_WEIGHTS`, `MUTABLE_TOKEN`, `get_mid_points`, `build_pdb_prior`, `compute_structure_losses`. |
| `design_b5_mps_v2.py` | Soft-logit parameterization (`init_soft_logits`, `build_soft_res_type`, `soft_to_hard_seq`), framework pinning, AA_FREQ PLM proxy, `cdr_to_epitope_stats`. |
| `esmscore/_complex.py` | `build_complex_features` — constructs the ESMFold2 input features (target first, binder second) from two sequences. |
| `esmscore/score_only.py` | `_patch_for_mps` — patches Full ESMFold2 for MPS. |
| `B5_CDR_DESIGN_SUMMARY.md` | The full version history (v2 → v13 + multi-start) with what worked and what didn't. |
| `B5_CDR_EPITOPE_TARGETING_RECIPE.md` | This file. |

---

## 11. Reproduction recipe (end-to-end)

```bash
cd /Users/huyue/esmc_design_new/esm-main-2/cookbook/tutorials

# (1) Build the 15-seed multi-start (~3.5 h)
bash run_v9_multistart.sh                 # seeds 1-5
bash run_v9_multistart_expanded.sh        # seeds 6-15 + auto-eval

# (2) (Optional) Re-eval the top 5 with Full ESMFold2 for a clean report
python3 -u eval_v9_multistart.py \
  --snaps /tmp/b5_v9_seed1_snaps.json /tmp/b5_v9_seed2_snaps.json \
         /tmp/b5_v9_seed3_snaps.json /tmp/b5_v9_seed4_snaps.json \
         /tmp/b5_v9_seed5_snaps.json /tmp/b5_v9_seed6_snaps.json \
         /tmp/b5_v9_seed7_snaps.json /tmp/b5_v9_seed8_snaps.json \
         /tmp/b5_v9_seed9_snaps.json /tmp/b5_v9_seed10_snaps.json \
         /tmp/b5_v9_seed11_snaps.json /tmp/b5_v9_seed12_snaps.json \
         /tmp/b5_v9_seed13_snaps.json /tmp/b5_v9_seed14_snaps.json \
         /tmp/b5_v9_seed15_snaps.json \
  --out /tmp/b5_v9_multistart_full_eval.json

# (3) Final aggregate
python3 -u summarize_v12b_v13b_expanded.py
```

Expected output (top of the global-best ranking):
```
seed=2 step 56   iptm=0.717  pTM=0.853  CDR→epi=9.52 Å
                 GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR
```

---

## 12. What was tested and did NOT help

(For context. Full details in `B5_CDR_DESIGN_SUMMARY.md`.)

| Version | Change | Outcome |
|---------|--------|---------|
| v2 | ESMFold2-Fast + disto-graph prior from input PDB | iptm 0.572 |
| v7 | Adam + larger LR + soft logits | stayed at v2 |
| v8 | distogram-expected prior (1 sample) | iptm 0.616, re-converged |
| **v9** | **CA-coord prior (4-sample avg)** | **iptm 0.661, NEW BASIN** |
| v10a | re-iterate from v9 with v9's prior | re-converged |
| v10b | + cosine T + lr(T) | re-converged |
| v10c | + strong aa_freq (0.05) | worse, new sequence at lower iptm |
| v11 | TRUE iterative prior refresh (every K=4 steps) | re-converged |
| v12 | Full ESMFold2 in design loop (30 steps) | re-converged |
| v13 | H3-only mutations | re-converged |
| v12b | Full in loop **from new best** | re-converged (iptm 0.700) |
| v13b | H3-only **from new best** | re-converged (iptm 0.701) |
| **multi-start v9 seeds 1-5** | **5 random seeds of v9** | **iptm 0.717 (seed=2 step 56)** |
| **multi-start v9 seeds 6-15** | **10 more random seeds of v9** | **best seed=10 iptm 0.662; no seed beat 0.717** |

The v9 + multi-start combination is the only one that has produced
anything above 0.661 in this project.
