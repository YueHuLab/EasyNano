# PAE vs ipSAE — quick reference

> **Short answer: NO, they are not the same thing.** PAE is the raw
> model output (a [L, L] matrix in Å). ipSAE is a *derived scalar
> metric* computed from a submatrix of PAE. They come from the same
> data but answer different questions.

---

## 1. The 30-second summary

| | PAE | ipSAE |
|---|---|---|
| **What** | raw model output | derived summary of PAE |
| **Shape** | [L, L] matrix | scalar |
| **Units** | Å (predicted error) | Å (typically) or unitless |
| **Range** | 0 to ~30 | 0 to ~30 (or 0-1 with d0 form) |
| **Source** | ESMFold2's PAE head | computed in post-processing |
| **Higher = ?** | worse | worse |
| **B5 example** | `pae.npy` shape [350, 350] | `ipsae_min_cdr_epi` ≈ 8-15 Å |

**The relationship is like**: PAE is the *raw data*, ipSAE is a
*chart you draw from the data*.

---

## 2. PAE — what it really is

`PAE[i, j]` = "If I align the predicted structure using residue j as
the reference frame, what would the expected position error of
residue i be, on average?"

- **Per pair** (one value per i, j pair)
- **Uncertainty measure**, not distance
- Shape: [L, L] where L = total residues (350 for B5: 127 binder +
  223 target)
- Tells you: "is the model confident about this pair's relative
  position?"
- **Does NOT** tell you: "what's the predicted distance between
  these two residues" (that's the distogram, a separate output)

A pair with PAE = 4 Å: model is *confident* in their relative
position. A pair with PAE = 25 Å: model is *clueless*.

---

## 3. ipSAE — what it really is

ipSAE = "the PAE of the contacts you care about, summarized somehow".

Standard definitions:
```
# Linear version (units = Å)
ipSAE_linear = mean(PAE[i, j] for selected pairs)
ipSAE_min   = min(PAE[i, j] for selected pairs)
ipSAE_p10   = 10th percentile of PAE[i, j] for selected pairs

# d0 version (unitless, 0-1)
ipSAE_d0 = mean( 1 / (1 + (PAE[i, j]/d0)^2) for selected pairs )
```

**"Interface" ipSAE** means: restrict the PAE pairs to (binder, target)
or (CDR, epitope). The "i" in ipSAE = "interface".

The "selected pairs" is everything — that's what makes ipSAE a
*summary* of PAE rather than PAE itself.

### The variants (p10, p15, etc.)
The `pX` variants (e.g., p10 = 10th percentile) are robust to
outliers:
- **`min`** = the single most confident pair (sensitive to noise)
- **`p10`** = top-10% most confident pairs (robust)
- **`p15`** = top-15% (more robust)
- **`mean`** = average (sensitive to bad pairs)

For binding: **`p10` or `p15` is the most useful variant** because
real binding is a few specific contacts, not all-pairs contact.

---

## 4. Concrete B5 example (real numbers)

For `v9_step48` (a known good binder, known iptm 0.619):
```
PAE:                     350 × 350 matrix  (saved as v9_step48_pae.npy)
PAE[CDR rows, epi cols]: 32 × 21 submatrix
   min  = 8.76 Å
   p10  = 10.42 Å
   mean = 15.50 Å
PAE[binder rows, target cols]: 127 × 223 submatrix (full binder×target)
   min  = 8.75 Å
   mean = 14.85 Å
```

The PAE submatrix `PAE[CDR, epi]` is 32 × 21 = 672 numbers. The
ipSAE values above are 3 different summaries of those 672 numbers.

---

## 5. Why the confusion exists

The naming is bad:
- **PAE** = "Predicted Aligned Error" — sounds like a single error
- **ipSAE** = "interface predicted Structural Alignment Error" — the
  "S" is silent but it's still PAE underneath

ipSAE is **literally a function of PAE**. If you have PAE, you can
compute ipSAE. If you have ipSAE, you can NOT recover PAE.

In AlphaFold2 paper terminology:
- **PAE** = "the [L, L] matrix the model outputs"
- **ipSAE** (Dunbrack convention, ~2023) = a *post-hoc* metric that
  became popular for evaluating designed interfaces

The AlphaFold2 paper doesn't actually use the term "ipSAE" — it's
a community convention for evaluating binder designs.

---

## 6. Other related metrics that are also NOT the same as PAE

| Metric | Source | Shape | What |
|---|---|---|---|
| **pTM** | TM-score head | scalar | predicted TM-score of whole complex |
| **ipTM** | TM-score head | scalar | predicted TM-score of interface |
| **chain_pair_iptm** | per-chain-pair head | [n_chains, n_chains] | chain-pair predicted TM-score |
| **pLDDT** | per-residue confidence head | [L] | per-residue predicted lDDT |
| **PAE** | PAE head | [L, L] | per-pair predicted aligned error |
| **ipSAE** | post-hoc from PAE | scalar | summary of PAE for selected pairs |

**None of these is the same as PAE**. They all come from different
heads (or post-processing). The only one derived from PAE is ipSAE.

For binding evaluation, **ipTM is the master metric**. ipSAE is a
useful secondary check on specific contacts.

---

## 7. When to use which

| You want to know... | Use |
|---|---|
| Is this sequence a good binder? | **ipTM** (median across n≥9 samples) |
| Are there specific confident contacts? | **ipSAE_p10(CDR, epitope)** |
| What's the predicted distance between i and j? | **distogram** (separate output) |
| Is the overall complex well-folded? | **pTM** (often uninformative for binding) |
| Is this single pair confidently placed? | **PAE[i, j]** |
| Which chains see each other? | **chain_pair_iptm** |
| Per-residue fold confidence? | **pLDDT** |

For final answer ranking, the B5 project uses:
1. **Primary**: median(ipTM) across 3 seeds × 3 samples = 9
2. **Secondary**: ipSAE_p10(CDR, epi) cross-check
3. **Tertiary**: std(ipTM) for basin stability

---

## 8. The key thing to remember

**PAE is a matrix. ipSAE is a scalar that summarizes a subset of
that matrix.** They are not the same thing. ipSAE requires choosing
which PAE pairs to include and how to summarize them.

- PAE[CDR, epi] is a 32×21 submatrix
- ipSAE_min(CDR, epi) is the single smallest value in that 32×21 submatrix
- ipSAE_p10(CDR, epi) is the 10th percentile of that 32×21 submatrix

Same data, different summary statistic.

---

## 9. Common B5 data showing the difference

| name | known_iptm | fold_iptm | PAE_min(CDR,epi) | PAE_p10(CDR,epi) | PAE_mean(CDR,epi) |
|---|---|---|---|---|---|
| v9_best_15seed | 0.717 | 0.397 | 11.92 | 14.40 | 18.25 |
| v9_step48 | 0.619 | 0.551 | 8.76 | 10.42 | 15.50 |
| v16_s5_s44 | 0.572 | 0.591 | 7.11 | 9.74 | 13.92 |
| v16_s5_s56 | 0.539 | 0.491 | 11.24 | 13.17 | 16.00 |
| v16_init_v2s050 | 0.471 | 0.477 | 14.14 | 16.19 | 19.08 |
| v16_s4_s44 | 0.104 | 0.137 | 25.13 | 28.39 | 31.21 |

**Note**: v9_best_15seed (highest iptm!) has *worse* ipSAE values
than v9_step48 (lower iptm). The two metrics disagree because they
measure different things:
- **ipTM** = "is the interface a coherent binding site?"
- **ipSAE** = "are there individual confident contact pairs?"

A design can have confident individual contacts (low ipSAE) but a
non-coherent interface (low ipTM), or vice versa. For B5 binding
quality, **ipTM is the more reliable metric**; ipSAE is a useful
diagnostic for "where exactly is the contact".

---

## 10. One-line summary

**PAE is the raw [L, L] matrix in Å. ipSAE is a scalar summary of a
selected submatrix of PAE. ipTM and pTM are completely separate
model outputs from different heads. For ranking B5 designs, use
ipTM (median over n≥9), not ipSAE.**
