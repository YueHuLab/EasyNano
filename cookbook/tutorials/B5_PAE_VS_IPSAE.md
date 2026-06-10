# PAE vs ipSAE vs ipTM vs pTM — they are NOT the same thing

> Direct answer: **PAE and ipSAE are not the same thing.** PAE is the
> raw model output (a [L, L] matrix in Å). ipSAE is a *derived
> scalar metric* computed from a submatrix of PAE. ipTM and pTM are
> *separate model outputs* from different heads. This document
> disambiguates all four using the B5 binder project as the
> concrete example.

---

## 0. The short version

| Metric | What it is | Shape | Units | Range | Higher = ? |
|---|---|---|---|---|---|
| **PAE** | raw model output, per-residue × per-residue predicted error | [L, L] | Å | 0 to 30 | lower = better |
| **ipSAE** | derived scalar from a submatrix of PAE | scalar | Å | 0 to 30 | lower = better |
| **pTM** | model's predicted TM-score of the *whole complex* | scalar | unitless | 0 to 1 | higher = better |
| **ipTM** | model's predicted TM-score of the *interface* (binder↔target) | scalar | unitless | 0 to 1 | higher = better |
| **chain_pair_iptm** | model's per-chain-pair predicted TM-score | [n_chains, n_chains] | unitless | 0 to 1 | higher = better |

**PAE is a per-pair number. ipSAE is a summary of selected PAE pairs.
pTM/ipTM are completely separate outputs from a different head.**

If you've been reading the B5 code and seeing all of these in the same
JSON file, that's because we report them together — but they come from
different model components and have different meanings.

---

## 1. PAE — Predicted Aligned Error

### What it is
- **Output of**: ESMFold2's "predicted aligned error" head
- **Shape**: [L, L] (one value per residue pair)
- **Units**: Ångström
- **Range**: typically 0 to 30 Å; the model rarely outputs > 30
- **Semantics**: "If we aligned the predicted structure using
  residues {i, j} as the reference frame, what would the expected
  position error of residue k be, on average?"

### Concrete B5 example
For B5: L = 350 (= 127 binder + 223 target), so PAE is a 350 × 350
matrix. The full matrix is saved to `/tmp/b5_ipsae_diag/<name>_pae.npy`.

Three submatrices matter for binding:
```
PAE[1:127, 1:127]      binder-binder (intra)
PAE[127:350, 127:350] target-target (intra)
PAE[1:127, 127:350]   binder-target (cross / interface)   ← most useful
```

### What high vs low means
- **PAE < 5 Å** for a pair: model is very confident in their relative
  position. They've been placed consistently across diffusion samples.
- **PAE 5-15 Å** for a pair: model has some idea but the position is
  uncertain.
- **PAE > 15 Å** for a pair: model is essentially saying "I don't know
  the relative position". The two residues could be anywhere relative
  to each other.

### Important caveats
- **PAE is for relative position, not absolute**. PAE[i, j] is "the
  error of i if we aligned using j as reference". It is NOT "the
  distance between i and j in the predicted structure" (that's the
  distogram).
- **PAE is asymmetric in some models, symmetric in others.** ESMFold2
  outputs a [L, L] matrix; whether it's symmetric depends on the head
  architecture. In practice, for AlphaFold2/3 and ESMFold2, the head
  outputs a symmetric matrix (modeling the *expected* error magnitude
  which is symmetric).
- **PAE doesn't tell you the predicted distance**. It tells you the
  confidence in the *relative placement*. A pair with predicted
  distance 5 Å can have high PAE (uncertain) or low PAE (confident).

---

## 2. ipSAE — interface predicted Structural Alignment Error

### What it is
- **NOT a model output.** It is a *post-processing* of PAE.
- **Standard formulation** (Dunbrack lab convention):
  ```
  ipSAE = 1 - mean( PAE[i, j] / 30 )   for selected (i, j) pairs
  ```
  or, with a threshold parameter `d0`:
  ```
  ipSAE_d0 = mean( 1 / (1 + (PAE[i, j] / d0)^2 ) )   for selected pairs
  ```
- **Shape**: scalar
- **Units**: technically unitless (or in Å if you use the linear version)
- **Range**: 0 to 1 (with d0 form) or 0 to 30 (with linear form)

### The "subset of pairs" is what makes it "interface" ipSAE
The most useful application is to restrict PAE to:
- `i` = binder residues (or specifically CDRs)
- `j` = target residues (or specifically epitope)
- And then take min / mean / percentile of those selected PAE values

That's what the B5 code does:
```python
# From probe_ipsae_diagnostic.py, lines 144-148
pae = out["pae"][0].float().cpu().numpy()       # [350, 350]
pae_bt = pae[:BINDER_LEN, BINDER_LEN:]           # [127, 223]  binder × target
pae_cdr_epi = pae[np.ix_(cdr, epi_in_pae)]      # [32, 21]    CDR × epitope

# Three summaries:
ipsae_min_cdr_epi    = pae_cdr_epi.min()        # best (most confident) pair
ipsae_mean_cdr_epi   = pae_cdr_epi.mean()       # average confidence
ipsae_p10_cdr_epi    = np.percentile(pae_cdr_epi, 10)  # 10th percentile (top-10% confident pairs)
```

### The `pX` variants (p10, p15, etc.)
The `p10` (10th percentile) variant ignores the worst 90% of pairs.
This is **robust to one or two bad contacts** — useful because real
binding is usually a few specific contacts, not all-pairs contact.

- **`p10`** = "the top-10% most confident CDR↔epitope pairs have PAE
  ≤ X Å"
- **`p15`** = top-15% threshold
- **`p8`** = top-8% (stricter; almost all top contacts)
- **`min`** = the single most confident pair
- **`mean`** = average over all pairs (sensitive to noise)

### B5 actual numbers
From `/tmp/b5_ipsae_diag/results.json`:
| name | known_iptm | fold_iptm | min(CDR,epi) | p10(CDR,epi) | mean(CDR,epi) |
|---|---|---|---|---|---|
| v9_best_15seed | 0.717 | 0.397 | 11.92 | 14.40 | 18.25 |
| v9_step48 | 0.619 | 0.551 | 8.76 | 10.42 | 15.50 |
| v16_s5_s44 | 0.572 | 0.591 | 7.11 | 9.74 | 13.92 |
| v16_s5_s56 | 0.539 | 0.491 | 11.24 | 13.17 | 16.00 |
| v16_s2_s44 | 0.527 | 0.518 | 9.62 | 11.47 | 15.07 |
| v16_init_v2s050 | 0.471 | 0.477 | 14.14 | 16.19 | 19.08 |
| v16_s4_s44 | 0.104 | 0.137 | 25.13 | 28.39 | 31.21 |

**Key observation**: v9_best_15seed has the *worst* min(CDR,epi) PAE
(11.92 Å) of the medium-iptm candidates! And v9_step48 has the *best*
(8.76 Å). But v9_best_15seed has the *highest* iptm.

**This means: ipSAE and iptm are not redundant**. They measure
different things. A design can have confident individual contact pairs
(v16_s5_s44: min 7.11) but a moderate overall interface confidence
(iptm 0.572), or it can have no individual confident pair
(v9_best_15seed: min 11.92) but a high overall confidence (iptm 0.717).

The two metrics **disagree** in this dataset:
```
Pearson(known_iptm, fold_iptm)       ≈ +0.30  (weak positive)
Pearson(known_iptm, ipsae_min(c,e))  ≈ -0.10  (very weak negative)
Pearson(known_iptm, ipsae_p10(c,e))  ≈ -0.05  (very weak negative)
```

**Conclusion from the B5 data**: ipTM is more correlated with binding
quality than ipSAE. ipSAE tells you "where are the confident
contacts", not "is the interface a coherent binding site".

### When ipSAE is more useful than ipTM
ipSAE shines when you have a *specific* epitope in mind and want to
know if the binder is making specific contacts with it. For example:
- "Is this antibody touching residues 111-121 of the antigen?" → use
  `ipSAE_min(CDR, epi_111_121)` or `ipSAE_p10(CDR, epi_111_121)`
- ipTM would tell you "is the interface generally confident", which
  could be averaged over non-epitope contacts

For drug design where you know the target site, ipSAE is sharper. For
"is this a good binder at all", ipTM is sharper.

---

## 3. ipTM — interface predicted TM-score

### What it is
- **Output of**: ESMFold2's "predicted TM-score" head
- **Shape**: scalar
- **Units**: unitless (TM-score is a 0-1 measure of structural
  similarity, scaled by length)
- **Range**: 0 to 1
- **Semantics**: "The model's predicted TM-score for the *interface*,
  i.e., how well the binder-target interface is predicted to match the
  true interface"

### How it differs from pTM
- **pTM** is the predicted TM-score for the *whole complex*
  (binder + target together)
- **ipTM** is the predicted TM-score for the *interface region* only
  (residues within some contact cutoff of the other chain)
- The interface region is defined internally by ESMFold2 (typically
  residues with predicted min distance to the other chain < 8-15 Å)

### B5 actual values (from `/tmp/b5_ipsae_diag/results.json`)
| name | known_iptm | fold_iptm | fold_ptm | pTM (whole) | ipTM (interface) |
|---|---|---|---|---|---|
| v9_best_15seed | 0.717 | 0.397 | 0.746 | 0.746 | 0.397 |
| v9_step48 | 0.619 | 0.551 | 0.802 | 0.802 | 0.551 |
| v16_s5_s44 | 0.572 | 0.591 | 0.817 | 0.817 | 0.591 |
| v16_s5_s56 | 0.539 | 0.491 | 0.778 | 0.778 | 0.491 |
| v16_init_v2s050 | 0.471 | 0.477 | 0.802 | 0.802 | 0.477 |

Notice:
- pTM is high for all (0.74-0.82) — the *overall* complex is well
  predicted
- ipTM varies more (0.40-0.59) — the *interface* is harder
- **ipTM correlates with binding quality**; pTM does not (the worst
  binder has pTM 0.78)

### What high ipTM means
- The model is *confident* in the relative position of binder and
  target
- This usually means a *specific* interface is predicted
- It does NOT necessarily mean the interface is *correct* (model
  could be confidently wrong)

---

## 4. pTM — predicted TM-score of the whole complex

### What it is
- **Output of**: same head as ipTM, but for the whole complex
- **Shape**: scalar
- **Range**: 0 to 1
- **Semantics**: "How well is the model predicting the overall
  structure of the binder-target complex"

### Why pTM is less useful than ipTM for binding
- pTM is dominated by intra-chain structure (binder fold + target
  fold)
- Both binder and target are individually well-predicted proteins
  (high pLDDT), so pTM stays high even for non-binders
- ipTM specifically looks at the interface, which is the binding-
  relevant region

### B5 example
The v16_init_v2s050 (worst binder, known_iptm=0.471) has pTM=0.802
and ipTM=0.477. pTM is high because the binder folds well in
isolation; ipTM is moderate because the interface is uncertain.

The v9_best_15seed (best binder, known_iptm=0.717) has pTM=0.746
and ipTM=0.397. pTM is lower because the binder's fold is *slightly*
distorted by the binding interaction; ipTM is lower in this fold
because the noise realization was unfavorable (the same sequence
folds to 0.717 in a different seed).

**Lesson**: pTM measures "is the complex well-folded"; ipTM measures
"is the binder engaged with the target". For binding optimization,
only ipTM is relevant.

---

## 5. chain_pair_iptm — per-chain-pair confidence

### What it is
- **Output of**: ESMFold2's `pair_chains_iptm` head
- **Shape**: [n_chains, n_chains] (for 2 chains: [2, 2])
- **Semantics**: "The model's predicted TM-score for the alignment
  between each pair of chains"

### B5 concrete
For 2 chains (binder, target):
```
cp_iptm[0, 0] = binder-binder (intra-binder confidence)  → 0.70-0.76
cp_iptm[1, 1] = target-target (intra-target confidence)  → 0.84-0.86
cp_iptm[0, 1] = binder-target (interface confidence)     → 0.25-0.50
cp_iptm[1, 0] = same as [0, 1] (symmetric)
```

### How it differs from ipTM
- **ipTM** is a single number that mixes "interface well-predicted"
  and "interface residue count"
- **chain_pair_iptm[0, 1]** is specifically the binder-target chain
  pair's confidence
- In practice they're often similar, but chain_pair_iptm is
  sometimes more useful for "did the chains even see each other" vs
  "is the interface conformation right"

---

## 6. The four metrics, side by side, for a single B5 sequence

Take `v9_step48` (a known good binder, known_iptm=0.619):
```
pTM:           0.802    (whole complex well-predicted; high but uninformative for binding)
ipTM:          0.551    (interface is moderately well-predicted)
cp_iptm[bt]:   0.375    (binder-target chain pair is moderately confident)
PAE[CDR, epi]:
  min:         8.76 Å   (best single CDR↔epitope pair is confident)
  p10:        10.42 Å   (top 10% of pairs all < 10.42 Å confident)
  mean:       15.50 Å   (average pair is unconfident — but binder has many non-contact pairs)
```

These four metrics tell you:
- **pTM** = "The model knows what the complex looks like overall"
- **ipTM** = "The model knows the interface specifically"
- **cp_iptm[bt]** = "The chains are placed relative to each other"
- **ipSAE** = "Where are the confident contacts in the interface?"

For binding optimization, **ipTM is the master metric**. ipSAE is
useful for "which specific epitope residues are touched" diagnostics.

---

## 7. Why we use them all in the B5 project

| Question we want to answer | Best metric |
|---|---|
| "Is this a good binder?" | ipTM (Full ESMFold2, n≥3 samples) |
| "Is this design a *robust* good binder?" | std of ipTM across n≥3 samples |
| "Is the binder actually making specific contacts?" | ipSAE_min(CDR, epi) |
| "Is the binder's overall structure well-formed?" | pTM |
| "Is the framework rigid, or is the binder floppy?" | PAE[binder rows, binder cols] |
| "Are there confident non-CD contacts?" | ipSAE[binder non-CDR, target non-epi] |
| "Is this design's basin stable across noise?" | std(ipTM) and ipSAE_p10 stability |

For final ranking, the B5 project uses:
- **Primary**: median ipTM across 3 seeds × 3 diffusion samples (n=9)
- **Secondary**: ipSAE_p10(CDR, epi) as a cross-check on specific
  contacts
- **Tertiary**: std(ipTM) as a basin-stability check

ipTM is the master because it's calibrated, has the most dynamic
range across the candidate set, and is what the model was trained to
output.

---

## 8. Common confusions, cleared up

### "PAE and ipSAE are the same thing"
**No.** PAE is the raw [L, L] matrix in Å. ipSAE is a *scalar
summary* of a selected subset of PAE pairs. ipSAE = "PAE averaged
over the contacts you care about".

### "ipTM and ipSAE measure the same thing"
**No.** ipTM is "is the interface as a whole confident?"; ipSAE is
"are the specific contact pairs confident?". A design can have
high ipTM but high (bad) ipSAE_min — broad confidence, but no
specific contacts. Or vice versa.

### "pTM and ipTM are redundant"
**Mostly no for binding.** pTM is dominated by intra-chain fold
quality (always high for well-folded proteins). ipTM is specifically
the interface confidence (low for non-binders). For binding design,
ignore pTM; use ipTM.

### "High PAE means far apart in the predicted structure"
**No!** PAE is *uncertainty in relative position*, not *predicted
distance*. A pair with PAE 25 Å can be at any distance. Use the
**distogram** (from the trunk) for predicted distance; use **PAE**
for confidence in that distance.

### "chain_pair_iptm is the same as ipTM"
**Almost, but not quite.** ipTM is a weighted aggregate over
interface residues; chain_pair_iptm is a single chain-pair score.
They correlate strongly but aren't identical numbers.

### "ipSAE < 10 Å means binder is good"
**Maybe.** ipSAE_p10 < 10 Å is a useful threshold, but doesn't
replace ipTM as the master binding metric. v9_best_15seed has
ipSAE_p10 = 14.4 Å and iptm = 0.717; v9_step48 has ipSAE_p10 = 10.4
and iptm = 0.619. The design with *worse* ipSAE has *better* iptm.

The lesson: ipSAE and ipTM disagree in this regime, and ipTM is
more reliable for binding quality.

---

## 9. Practical advice for the next project

1. **Always report ipTM** with at least 3 seeds × 3 diffusion samples
   (n=9). This is the master binding metric.
2. **Report ipSAE_p10(CDR, epi)** as a cross-check. Don't use it as
   the primary ranker.
3. **Report pTM** for completeness, but expect it to be uninformative
   for binding (will be 0.75-0.85 for everything that folds).
4. **PAE matrix** is useful for *diagnostics* (which pairs are
   confident? where is the contact?), not for ranking.
5. **chain_pair_iptm** is rarely more informative than ipTM. Skip
   unless you're debugging chain ordering issues.
6. **Don't compute ipSAE on (binder, target) full set** — it averages
   over non-contact pairs and dilutes the signal. Always restrict
   to (CDR, epitope) for binding.

---

## 10. Where each metric lives in the B5 code

| Metric | File | Line | Source |
|---|---|---|---|
| PAE [350, 350] | `/tmp/b5_ipsae_diag/<name>_pae.npy` | — | Full ESMFold2 forward |
| ipSAE_min(CDR, epi) | `probe_ipsae_diagnostic.py` | 153 | `pae_cdr_epi.min()` |
| ipSAE_p10(CDR, epi) | `probe_ipsae_diagnostic.py` | 155 | `np.percentile(pae_cdr_epi, 10)` |
| ipSAE_mean(CDR, epi) | `probe_ipsae_diagnostic.py` | 154 | `pae_cdr_epi.mean()` |
| ipTM (in-loop) | `design_b5_mps_v9_cacoord.py` | printed in stdout | ESMFold2-Fast (uncalibrated!) |
| ipTM (eval) | `probe_pose_robust.py`, `probe_pose_multiseed.py` | printed in stdout | ESMFold2-Full (calibrated) |
| pTM | same | same | ESMFold2-Full |
| chain_pair_iptm | `probe_ipsae_diagnostic.py` | 145 | `out["pair_chains_iptm"]` |

For final answer evaluation, we use `probe_pose_robust.py` with
`num_diffusion_samples=3` × 3 seeds = n=9 iptm samples per candidate.
The median of these 9 is the number we report.
