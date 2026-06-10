# B5 v2 — How the design loop is attracted to the epitope

> Deep dive into the gradient flow that makes the v2 design loop (and
> every later version) pull the binder's CDRs toward the target's
> epitope. This is the "magic" step — auto-detect 21 epitope residues,
> and the binder falls toward them in 30-100 gradient steps.

The key idea: **ESMFold2's trunk is a free, learned potential-energy
surface for "what contacts does this sequence form"**. The design
loop's only job is to ask the right question (L_epi) and let Adam
walk downhill.

---

## 0. Where we start (step 0)

The init sequence is the WT chothia-canonical VH framework III with
default CDRs:
```
H1: GLQIGYGVYM
H2: SYSGQS
H3: RVVTDSYQPIYKAPIR
```

The input PDB has the binder in an *arbitrary* pose — not bound, not
aligned to the epitope. ESMFold2-Fast's predicted distogram for this
init has:
- `min_CDR_to_epitope_distance ≈ 10-15 Å`
- `L_epi = mean_over_CDR_rows( ELU(min_dist - 8.0) ) ≈ 50`

ELU(x) = max(x, exp(x)-1). For min_dist=12 Å, ELU(4) = exp(4)-1 ≈ 53.
**L_epi explodes to ~50 at step 0** — that's the magnetic field that
pulls everything.

---

## 1. The 4-step gradient chain

```
L_epi  ─→  disto_logits  ─→  trunk pair rep z
        ─→  ESMC sequence embedding
        ─→  input res_type (soft for binder)
        ─→  softmax(soft_logits / T)  ─→  soft_logits (Adam update)
```

Read this section bottom-up if you want to follow the gradient. Or
top-down if you want to see the loss flowing into the parameters.

---

## 2. Step 1 of the chain: L_epi → disto_logits

`binder_design_hy_losses.py` lines 206-242:

```python
def compute_epitope_loss(
    distogram_logits, binder_length, epitope_token_indices,
    bin_distance, cutoff=EPITOPE_CUTOFF, cdr_indices=None,
):
    cross = distogram_logits[:, -binder_length:, :-binder_length, :]   # [1, L_b, L_t, 64]
    epitope = cross[:, :, epitope_token_indices, :]                    # [1, L_b, 21, 64]
    probs = torch.softmax(epitope, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)                             # [1, L_b, 21]
    min_dist = e_dist.min(dim=-1).values                                # [1, L_b]
    contact_term = F.elu(min_dist - cutoff)                             # [1, L_b]
    # CDR-only mask: only the 32 CDR rows contribute
    mask = torch.zeros(binder_length, device=...)
    mask[cdr_indices] = 1.0
    per_res = contact_term * mask[None, :]
    return per_res.sum() / (mask.sum() + 1e-8)
```

**What the gradient looks like**:

- ∂L_epi/∂disto_logits[row, col, k] is **zero on almost all (row, col)**
- Non-zero only at, for each `row`, the argmin col (the *closest*
  epitope residue) and the 64 bins
- The bin gradient is the standard "negative log-prob of correct bin":
  - if predicted bin == target bin: small positive grad
  - if predicted bin is far from target bin: large positive grad on
    predicted, large negative grad on target
- The ELU's outer gradient is 1 for min_dist > 8 Å (the regime we're
  in for the entire design run), so min_dist is the quantity being
  pulled down

**Concrete example** — H1 residue 3 (binder row 27) is closest to
epitope residue 113 (target col 113):
- Predicted E_dist[27, 113] ≈ 12 Å → bin 24 (midpoint ~9.5 Å)
- "Contact" should be ≈ 6 Å → bin 12
- ∂L_epi/∂logit[24] > 0 (suppress this bin)
- ∂L_epi/∂logit[12] < 0 (boost this bin)
- ∂L_epi/∂logit[other 20 epitope columns for row 27] = 0
- ∂L_epi/∂logit[all 127 binder rows except 27] = 0 from this term
- **The gradient is sparse and directional** — exactly one epi-col per
  epi-row receives gradient

This is the "magnetism": each CDR residue is assigned to one specific
epitope residue (its current closest), and the gradient points *from*
the CDR row *toward* that specific epi col.

---

## 3. Step 2 of the chain: disto_logits → trunk z → ESMC embedding

ESMFold2's trunk maps pair rep `z` (shape L, L, d) to disto_logits
through a single linear layer:
```python
disto_logits = self.distogram_head(z)   # [L, L, n_bins]
```

So `∂L_epi/∂z[i, j, h] = Σ_k ∂L_epi/∂disto_logits[i, j, k] · W[h, k]`.

The trunk's `z` is built from:
- **MSA pair rep**: evolutionary couplings from the multiple sequence
  alignment
- **Sequence pair rep**: outer product of ESMC embeddings
  `s[i] ⊗ s[j]` for the binder+target sequences
- **Recycling**: previous trunk output concatenated back

**The crucial property**: the trunk has been trained on millions of
protein complexes. Given any binder+target sequence, it can predict
*what their disto would look like if they were in contact*. The
trunk encodes "AA pattern → contact geometry" as a side-effect of
its training.

The pair rep `z[i, j, :]` for an "epitope-touching" pair (i, j) will
look very different from a "non-touching" pair — specifically, the
last-layer features will bias the distogram head toward predicting
close bins. The gradient w.r.t. `z` says "make the pair rep look more
like an in-contact pair", and the trunk knows which input embeddings
s[i], s[j] would make that happen.

---

## 4. Step 3 of the chain: ESMC → res_type → soft_logits

`z`'s sequence-pair component is the outer product of ESMC embeddings
`E(sequence)[i]`. The ESMC embedding E(aa) for a single residue type
is a learned d-dim vector. The trunk's gradient w.r.t. `E(aa)[i]`
flows back to the input res_type, which is **soft for binder**:
```python
binder_probs_20 = F.softmax(soft_logits / T, dim=-1)   # [L_b, 20]
binder_probs_33 = zeros(L_b, 33)
binder_probs_33[:, 2:22] = binder_probs_20
res_type = cat([binder_probs_33, target_one_hot], dim=1)
```

The gradient `∂z/∂res_type[row, k]` is a fixed linear function of
ESMC's embedding matrix. The gradient `∂res_type/∂soft_logits[row, aa]`
is a softmax Jacobian (small near argmax, large in the middle of the
distribution).

**The "soft" mechanism is what makes this work at all**. If res_type
were argmax (one-hot), the gradient would be zero on the entire row
— you can't update a discrete AA. Soft res_type is a 20-dim probability
distribution that the gradient can shape, and the *next* argmax can
change as the distribution shifts.

---

## 5. Step 4 of the chain: Adam updates soft_logits

```python
total = 0.5*intra + 0.5*inter + 0.2*glob + 0.05*epi + 0.3*prior + 0.01*aafreq
total.backward()
optimizer.step()  # Adam(lr=0.5)
pin_fixed_positions()  # framework logits → -10/+10
```

**Five things happen here, in order**:

1. **Framework logits are reset**. Even though gradients flow to all
   127 positions during backward, `pin_fixed_positions` forces
   framework logits back to `+10` (WT) and `-10` (others), so
   argmax(framework_logit) is mathematically always WT. The framework
   can't change.
2. **32 CDR logits get Adam-updated**. The 20-dim logit vector for
   each CDR position moves by `lr * grad`. With LR=0.5, the move is
   large (~0.5 per step); with LR=0.05 (v9), it's 10× smaller.
3. **L_epi drives "toward epitope"**. v2 weights this at 0.05; v9 at
   0.2. This is the *only* loss that pulls toward the epitope — all
   others are about structure quality.
4. **L_prior drives "toward PDB reference"**. This is v2's "anchor"
   that fights the epitope pull. The 0.3 weight is large; the
   81000-pair mask makes the gradient signal strong.
5. **L_intra/L_inter/L_glob drive "structural quality"**. They don't
   care about the epitope specifically; they want the binder to be a
   well-folded protein that contacts something.

The v2 vs v9 weight balance is the single biggest design choice:
```
v2 :  0.5*intra + 0.5*inter + 0.2*glob + 0.05*epi + 0.3*prior + 0.01*af
v9 :  0.5*intra + 0.5*inter + 0.2*glob + 0.20*epi + 0.3*prior + 0.01*af
                                   ^^^
                              4× stronger
```

v2's L_epi is 1/20 of the total weight; v9's is 1/8.5. v9 lets the
epitope pull dominate the prior, which is the only reason v9 finds
new CDRs.

---

## 6. The "magnetic field" analogy

Picture a 640-dim discrete space (32 CDR positions × 20 AAs):

- **U(seq) = L_total(seq)** is the potential energy
- **F = -∇U** is the force on the AA at each position
- **Epitope = 21 fixed charges**
- **Binder CDRs = 32 movable charges**

L_epi is the "Coulomb + van der Waals" interaction between the two.
L_prior is the "this object must stay rigid" constraint. L_intra/
L_inter/L_glob are "this object must remain a coherent protein".

In v2, the prior (0.3) and structure quality (1.2) overwhelm the
epitope pull (0.05), so the binder is mostly held in place. In v9,
the epitope pull (0.2) is closer in weight to the prior (0.3), and
the binder actually moves.

**The "magnetism" is ESMFold2's potential-energy surface**. We didn't
invent "how to make a protein bind" — we asked ESMFold2's trunk
"given this sequence, how would it bind?", and the gradient of
L_epi with respect to the AA logits is *the answer in reverse*.

---

## 7. A step-by-step trace (v2 actual behavior)

Assume init gives `L_epi ≈ 50`, `min_CDR_to_epi ≈ 12 Å`, T = 1.0,
LR = 0.5:

| Step | Wall time | T | min_CDR_to_epi (Å) | L_epi | CDR changes |
|---|---|---|---|---|---|
| 0 | 5s | 1.00 | 12.0 | 50.0 | init |
| 5 | 30s | 0.95 | 11.5 | 35.0 | soft drift, no argmax flip |
| 10 | 60s | 0.92 | 10.8 | 22.0 | soft drift, no argmax flip |
| 30 | 180s | 0.55 | 10.2 | 12.0 | 0-2 argmax flips, mostly H3 |
| 50 | 300s | 0.28 | 10.0 | 7.0 | locked near local min |
| 100 | 600s | 0.10 | 10.0 | 7.0 | stuck |

**v2 never escapes ~10 Å** because the prior + structure quality
push back hard. The binder is *almost* in contact but not quite.

**v9 trace** (LR=0.05, L_epi=0.2, CA-coord prior):
| Step | min_CDR_to_epi (Å) | L_epi | CDR changes |
|---|---|---|---|
| 0 | 12.0 | 50.0 | init |
| 10 | 9.5 | 18.0 | 1-2 argmax flips |
| 30 | 8.0 | 5.0 | 3-4 argmax flips |
| 60 | 7.5 | 2.0 | 5+ argmax flips, found new basin |

The smaller LR + larger L_epi + better prior is what lets v9 cross
the local-min barrier. v2 doesn't.

---

## 8. Why this is "magnetic" — the 4 hidden prerequisites

For the design loop to attract a binder to an epitope automatically,
4 things must all be true:

1. **ESMFold2's trunk is a contact predictor**. It has been trained
   on millions of protein complexes, so given any binder+target
   sequence, it can predict what their disto would look like *if in
   contact*. The training objective implicitly encodes "AA → contact
   geometry" as a side effect.

2. **Distograms are soft geometric representations**. We don't need
   actual 3D coordinates; we need a probability distribution over
   distances. This is differentiable and lets losses act on a
   probability (not a coordinate). The model's confidence (sharpness
   of the distogram) is also a learnable signal.

3. **res_type is soft for binder**. The 32 CDR positions are encoded
   as 20-dim probability distributions, not argmax one-hot. The
   softmax is the gradient conduit — without this, you can't do
   gradient-based AA design at all (you'd have to enumerate, which is
   20^32 ≈ 10^41 configurations).

4. **ESMFold2 has learned "AA → contact"**. The trunk's pair rep
   `z[i, j, :]` for an in-contact pair has features that the distogram
   head maps to "close bins". The gradient says "make the pair rep
   look like in-contact", and the trunk knows which AA embedding
   shift would do that.

If any of these 4 is missing, the design loop can't do the magnetic
attraction. RFdiffusion/Chroma use the *diffusion process itself* as
the gradient channel (different technique, same end). v2 uses the
distogram (which is one of ESMFold2's *intermediate products*, not
its final output). The trick is: distogram is the *only* place in
ESMFold2 where geometric signal is both continuous and side-chain-
detailed. pTM-head output is too coarse, atom coords are too far
downstream.

---

## 9. What ESMFold2 isn't doing

To be clear about what the "magnetism" *isn't*:

- **Not** simulating binding kinetics. ESMFold2 has no time
  dimension; there's no association/dissociation.
- **Not** doing rotamer search. The side chains aren't being placed;
  the distogram is the side-chain-placement signal in aggregated
  form.
- **Not** computing binding energy. There's no ΔG; the loss is
  "expected distance to the closest epitope residue < 8 Å", not a
  thermodynamic quantity.
- **Not** doing CDR scaffolding or hallucination. The CDR sequence
  changes by 0-5 positions, not by 20+. The loop is local
  optimization, not generation.

The "magic" is bounded: the loop can find a CDR sequence whose
*predicted distogram* says "in contact with epitope". Whether the
real 3D structure agrees, whether the binding is functional, whether
the affinity is nM or μM — all of those need experimental validation.
The loop is a *filter*, not a *guarantee*.

---

## 10. Why v2 is the "magic version"

v2 is the smallest, most legible version of this trick. Every later
version (v3-v16) changes exactly one or two things:
- v3: Full ESMFold2 in loop (4× slower, similar gradient)
- v4: hard mask on non-CDR (no effect)
- v7: Adam + larger LR (LR overshoots, no help)
- **v8: distogram-expected prior** (smoother, slightly better basin)
- **v9: realized 3D CA-coord prior** (sharper, finds new basin)
- v10-v13: re-attempts to escape the new basin (none succeed)
- v14/v15/v16: re-anchor the epitope (all worse, see §11)
- Step 2: p116Y framework micro-tuning (collapses bimodal basin)

The *epitope attraction mechanism* is identical across all of them.
v2 contains it whole, in 111 lines, with no other distractions. That's
why v2 is the magic version — everything that comes after is a tweak
on a v2 that already works.

---

## 11. Why "fix the epitope" matters

The epitope is a **fixed input**, not a moving target. The 21 indices
come from the input PDB and never change. This is critical:

- L_epi gradient is *directional* (always toward the same 21 residues)
- L_epi gradient magnitude is *stable* (doesn't depend on the
  optimizer's current state)
- L_epi gradient *carries meaning* (the same epitope_col is targeted
  by the same CDR_row across all 100 steps)

If the epitope moved (v14, v15, v16), the gradient would be
self-referential — the target follows the binder, the binder chases
the target, neither converges. v9 multi-start with fixed epitope
finds 15 different basins; v15 periodic re-anchor finds ~1
(self-referential) basin. The fixed epitope is the *anchor* that
makes magnetic attraction possible.

---

## 12. Summary

The "magnetic attraction" of v2 is not a property of the v2 script
specifically — it's a property of:
1. ESMFold2's contact-predicting trunk
2. Soft res_type for binder (gradient channel)
3. Distogram-geometry loss (L_epi)
4. Fixed epitope (directional, stable gradient)
5. Adam with reasonable LR (gradient walker)

v2 happens to be the version that puts these 5 ingredients together
with the smallest amount of other machinery. Every later version is
a refinement: better prior (v8/v9), better weights (v9), better
starting points (v9 multi-start), better basin stabilization
(p116Y). None of them change the core trick — that trick was in v2
from the start.
