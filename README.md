# EasyNano

**Rapid epitope-targeted nanobody CDR design via differentiable distogram optimization with ESMFold2.**

EasyNano optimizes nanobody CDR sequences by backpropagating epitope proximity signals through the ESMFold2 distogram. It runs in **~10-20 minutes per target** on a high-end personal workstation (tested on Apple Mac Studio M3 Ultra, 256 GB).

> Hu, Y., Cheng, W., Wang, J., & Liu, Y. *EasyNano: rapid epitope-targeted nanobody CDR design via differentiable distogram optimization with ESMFold2.* (2026)

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers biotite abnumber numpy

# Set model paths
export ESMFOLD2_FAST=/path/to/ESMFold2-Fast    # 721M, for design loop
export ESMFOLD2_FULL=/path/to/ESMFold2         # 1.3B, for evaluation
export ESMC_PATH=/path/to/ESMC-6B              # 6B language model encoder

# All-in-one pipeline (5 targets from the manuscript)
easynano run --target ty1_rbd --seeds 0 1 2 --out-dir results/ty1_rbd

# Or step by step:
easynano setup --target ty1_rbd                                    # Stage 1: setup
easynano design --target ty1_rbd --seeds 0 1 2                    # Stage 2: design
easynano eval --target ty1_rbd --snapshots results/ty1_rbd/seed0_snapshots.json
easynano baseline --target ty1_rbd --n-random 30                   # Statistical test
```

**First run?** Start with the `setup` command to verify everything works:
```bash
easynano setup --target vhh72_rbd
```

---

## Pipeline Overview

```
Stage 1: Structure Prior (~30 s)
  Full ESMFold2 (1.3B) folds WT complex
  → Extracts CA-coordinate distance prior
  → Anchors framework pose

Stage 2: Differentiable CDR Design (~10-17 min)
  ESMFold2-Fast (721M) as differentiable oracle
  → CDR logits optimized via gradient descent through distogram
  → Composite loss: epitope + contacts + structure prior
  → 60 Adam steps, cosine temperature annealing

Stage 3: Full Model Evaluation (~15 s/candidate)
  Full ESMFold2 evaluates top candidates
  → Reports ipTM, pTM, CDR-to-epitope distance

Stage 4: Random Baseline (~15 min, n=30)
  Null distribution of random CDR ipTM
  → Statistical significance (zσ)
```

---

## Available Targets

| Name | Description | WT ipTM | Best Design ipTM | CDR len |
|------|-------------|---------|------------------|---------|
| `ty1_rbd` | Ty1 / SARS-CoV-2 RBD (6ZXN) | 0.143 | 0.702 | 22 |
| `kn035_pdl1` | KN035 / PD-L1 (5JDS) | 0.251 | 0.459 | 32 |
| `vhh72_rbd` | VHH72 / RBD (6WAQ) | 0.776 | 0.742 | 29 |
| `vhh3_tnfa` | VHH3 / TNFα (5M2M) | 0.672 | 0.671 | 33 |
| `antitnf_tnfa` | anti-TNF / TNFα (5M2J) | 0.207 | 0.231 | 19 |
| `b5_aqp4` | B5 / AQP4 (de novo) | 0.117 | 0.538 | 32 |

To add a custom target, use `--pdb`, `--chain`, and `--epitope-indices` instead of `--target`:
```bash
easynano design --pdb my_target.pdb --chain A \
    --epitope-indices "10,11,12,15,16,17" \
    --framework b5 --seeds 0 1 2
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wt-logit` | 2.0 | **Critical.** WT AA bias at init. Too high (≥5.0) = CDR locked. Too low (≤1.0) = chaotic. |
| `--w-prior` | 0.05 | Structure prior weight. Too high (≥0.3) = over-constrained. Too low (<0.05) = pose drift. |
| `--steps` | 60 | Optimization steps. 30 = bounce-back risk. 60 = stable convergence. |
| `--lr` | 0.05 | Adam learning rate |
| `--seeds` | [0] | Random seeds. **Use ≥3** — single-seed rankings are not significant. |
| `--allow-cdr-cys` | False | Permit cysteine in CDRs (required for KN035 which has a structural disulfide). |

These defaults are the "sweet spot" identified by systematic parameter sweep on RBD/VHH72 (6WAQ).

---

## Output Files

```
results/<target>/
├── seed0_snapshots.json    # Design trajectory (step, CDR, losses, ipTM)
├── seed1_snapshots.json
├── seed2_snapshots.json
├── eval_seed0.json          # Full ESMFold2 evaluation
├── eval_seed1.json
├── eval_seed2.json
└── random_baseline.json     # Random CDR ipTM distribution (n=30)
```

---

## Loss Functions

The composite loss optimized in the design loop:

| Loss | Weight | Purpose |
|------|--------|---------|
| Epitope | 0.2 | Pulls CDR residues toward epitope |
| Intra contact | 0.5 | Binder internal contacts (packing) |
| Inter contact | 0.5 | Binder-target interface contacts |
| Globularity | 0.2 | Compactness (radius of gyration) |
| Structure prior | 0.05 | Anchors framework pose |
| AA frequency | 0.01 | Keeps AA distribution natural |

---

## Developability Analysis

Check sequence properties for manufacturability:
```bash
easynano analyze --developability --sequence GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY
```

Reports: GRAVY (hydrophobicity), net charge, aromatic %, PTM risk sites (deamidation, oxidation, isomerization, proteolysis), and flag warnings.

---

## Model Paths

By default, paths point to a Mac Studio M3 Ultra setup. Override via environment variables:

```bash
export ESMFOLD2_FAST=/path/to/ESMFold2-Fast
export ESMFOLD2_FULL=/path/to/ESMFold2
export ESMC_PATH=/path/to/ESMC-6B
export EASYNANO_DEVICE=mps          # or cuda, cpu
```

Or edit `easynano/config.py` directly.

---

## Citing

```bibtex
@article{hu2026easynano,
  title   = {EasyNano: rapid epitope-targeted nanobody CDR design via
             differentiable distogram optimization with ESMFold2},
  author  = {Hu, Yue and Cheng, Wanyu and Wang, Junqing and Liu, Yingchao},
  journal = {bioRxiv},
  year    = {2026}
}
```

---

## Limitations

1. **No AF2/AF3 cross-validation built in** — the method relies entirely on ESMFold2. We recommend validating top designs with an orthogonal model (AF3 web server or local AF2).
2. **CDR length is fixed** — H3 length is not optimized.
3. **Framework is pinned** — only CDR residues are mutable.
4. **No experimental validation** — in silico ipTM does not guarantee actual binding. SPR/BLI confirmation is essential.
5. **Pose-dependent** — if ESMFold2's predicted pose is wrong (e.g., >20Å from crystal), CDR optimization cannot rescue it.

---

## Repository Structure

```
EasyNano/
├── easynano/
│   ├── __init__.py      # Version
│   ├── config.py         # All paths, constants, target registry
│   ├── loss.py           # Core loss functions
│   ├── setup.py          # Target PDB loading + epitope/CDR identification
│   ├── design.py         # V9 design loop (CA-coordinate prior + Adam optimization)
│   ├── evaluate.py       # Full ESMFold2 evaluation
│   ├── baseline.py       # Random CDR baseline (statistical significance)
│   ├── analyze.py        # Supplementary analyses (developability, Fast vs Full)
│   └── cli.py            # Unified CLI (easynano setup|design|eval|baseline|run)
├── cookbook/             # Legacy scripts, documentation, and experiments
├── test/                 # Target PDB files
└── README.md
```

---

## License

MIT
