# EasyNano: Epitope-Targeted Nanobody CDR Design with ESMFold2

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**EasyNano** is a practical pipeline for rapid, epitope-targeted nanobody complementarity-determining region (CDR) design powered by ESMFold2 differentiable distogram optimization. It runs in approximately 10--20 minutes per target on a high-end personal workstation (tested on Apple Mac Studio M3 Ultra, 256 GB unified memory).

> Hu, Y., Cheng, W., Wang, J., & Liu, Y. *EasyNano: rapid epitope-targeted nanobody CDR design via differentiable distogram optimization with ESMFold2.* (2026)

---

## Key Features

- **Epitope-specific**: targets user-specified epitope residues via a dedicated CDR-to-epitope distance loss
- **Fast**: ~10--20 minutes per target on a high-end personal workstation (Mac Studio M3 Ultra)
- **Pose-preserving**: Full ESMFold2 CA-coordinate structure prior prevents framework drift
- **De novo capable**: manual docking + EasyNano optimization enables design from scratch
- **Statistically validated**: random CDR baselines ($n=30$) confirm significance

## Results at a Glance

| Target | WT ipTM | Design ipTM | Improvement | Scenario |
|--------|---------|-------------|-------------|----------|
| Ty1 / SARS-CoV-2 RBD (6ZXN) | 0.143 | **0.702** | **+0.559** | Self-recovery |
| KN035 / PD-L1 (5JDS) | 0.251 | **0.459** | **+0.208** | Self-recovery |
| B5 / AQP4 | 0.117 | **0.538** | **4.6×** | De novo design |
| VHH72 / RBD (6WAQ) | 0.776 | 0.742 | −0.035 | Strong binder (safe) |
| VHH3 / TNFα (5M2M) | 0.672 | 0.671 | −0.001 | Strong binder (safe) |
| anti-TNF / TNFα (5M2J) | 0.207 | 0.231 | +0.024 | Limited by short CDR |

## Installation

```bash
# Clone the repository
git clone https://github.com/YueHuLab/EasyNano.git
cd EasyNano

# Install dependencies
pip install torch transformers biotite abnumber logomaker pandas matplotlib seaborn

# ESMFold2 models are expected at:
#   /path/to/ESMFold2-Fast   (721M, for design loop)
#   /path/to/ESMFold2        (1.3B, for prior + evaluation)
#   /path/to/ESMC-6B         (6B language model encoder)
# Update MODEL_PATH / ESMC_PATH in scripts to match your paths.
```

## Pipeline Overview

```
Stage 1: Structure Prior (~30 s)
  Full ESMFold2 (1.3B) folds WT framework+target complex
  → Extracts CA-coordinate distance prior
  → Anchors framework pose for optimization

Stage 2: Differentiable CDR Design (~10-17 min)
  ESMFold2-Fast (721M) serves as differentiable oracle
  → CDR logits optimized via gradient descent through distogram
  → Composite loss: epitope + intra/inter/glob contacts + structure prior
  → 60 Adam steps with cosine temperature annealing

Stage 3: Full Model Evaluation (~15 s/candidate)
  Full ESMFold2 evaluates top candidates
  → Reports ipTM, pTM, CDR-to-epitope distances
  → Random CDR baselines (n=30) establish null distributions
```

## Repository Structure

```
EasyNano/
├── cookbook/tutorials/
│   ├── design_ca_prior.py       # Main v9 design loop (CA-coordinate prior)
│   ├── design_target_prior.py   # v2 design loop (target-only prior)
│   ├── setup_target.py          # Target setup from arbitrary PDB + epitope
│   ├── loss_functions.py        # Core loss functions (epitope, prior, contacts)
│   ├── sweep_parameters.py      # Parameter sweep runner
│   ├── evaluate_designs.py      # Batch Full ESMFold2 evaluation
│   ├── evaluate_snapshots.py    # Per-target snapshot evaluation
│   ├── random_baseline.py       # Random CDR baseline generation
│   ├── validate_rmsd.py         # Kabsch RMSD cross-validation
│   ├── plot_figures.py          # Manuscript figure generation
│   ├── plot_cdr_logos.py        # CDR sequence logo generation
│   ├── analyze_losses.py        # Loss function analysis
│   └── run_designs_*.sh         # Batch design runners
├── manuscript/
│   ├── main.tex / main.pdf      # Preprint manuscript
│   ├── supplementary.*          # Supplementary information
│   └── figures/                 # All manuscript figures (PDF)
├── test/                        # Target PDB files (6 targets)
├── pyproject.toml
└── LICENSE
```

## Quick Start

```bash
# 1. Setup target from PDB
python cookbook/tutorials/setup_target.py \
    --target-pdb test/5JDS.pdb --target-chain A \
    --auto-detect-binder B --framework kn035

# 2. Run v9 design (CA-coordinate prior)
python cookbook/tutorials/design_ca_prior.py \
    --target-pdb test/5JDS.pdb --target-chain A \
    --epitope-indices "36,38,43,45,48,50,97,98,99,101,102,103,104,105" \
    --framework kn035 --wt-logit 2.0 --w-prior 0.05 \
    --steps 60 --seed 0 --allow-cdr-cys

# 3. Evaluate designs with Full ESMFold2
python cookbook/tutorials/evaluate_designs.py \
    --n-top 3 --out /tmp/eval_results.json

# 4. Run random baselines
python cookbook/tutorials/random_baseline.py \
    --target-pdb test/5JDS.pdb --target-chain A \
    --epitope-indices "36,38,43,45,48,50,97,98,99,101,102,103,104,105" \
    --framework kn035 --n-random 30 --out /tmp/baseline.json
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wt-logit` | 2.0 | WT amino acid logit bias at initialization (critical: too high = CDR locked, too low = chaotic) |
| `--w-prior` | 0.05 | Structure prior weight (anchors framework pose) |
| `--steps` | 60 | Optimization steps (30 = bounce-back, 60 = stable convergence) |
| `--lr` | 0.05 | Adam learning rate |
| `--seed` | 0 | Random seed (use ≥3 for robust results) |
| `--allow-cdr-cys` | False | Permit cysteine in CDRs (required for disulfide-containing frameworks like KN035) |

## Targets Evaluated

| Target | PDB | Framework | CDR length | Type |
|--------|-----|-----------|------------|------|
| Ty1 / SARS-CoV-2 RBD | 6ZXN | Ty1 | 22 | Self-recovery |
| KN035 / PD-L1 | 5JDS | KN035 | 32 | Self-recovery (disulfide) |
| VHH72 / SARS-CoV-2 RBD | 6WAQ | VHH72 | 29 | Strong binder control |
| VHH3 / TNFα | 5M2M | VHH3 | 33 | Strong binder control |
| anti-TNF / TNFα | 5M2J | anti-TNF | 19 | Short CDR limit case |
| B5 / AQP4 | B5 | B5 (framework III) | 32 | De novo design |

## How It Works

EasyNano optimizes CDR sequences **indirectly**: it does not optimize ipTM directly (which would require
prohibitively expensive Full ESMFold2 confidence-head evaluations at every gradient step). Instead, it
optimizes the **ESMFold2-Fast distogram**—a lightweight, differentiable proxy. The rationale is that
sequences producing distograms with favorable properties (low epitope distance, high contact confidence,
preserved pose) should also yield high ipTM when evaluated with the full model.

This "distogram-as-proxy" strategy was introduced in the official ESMFold2 binder design tutorial
(`binder_design.py`). EasyNano extends it with:
1. **Epitope targeting**: a dedicated CDR-to-epitope distance loss
2. **CA-coordinate structure prior**: locks the framework pose, enabling de novo design
3. **WT-biased initialization** (wt_logit = 2.0): balances exploration vs physicality

## Relationship to Official ESMFold2 Code

The official ESMFold2 codebase includes `binder_design.py`, which pioneered gradient-guided sequence
optimization through the distogram. We gratefully build upon this foundation. The official code performs
general binder design (optimizing a sequence to contact a target at *any* interface) without epitope-level
targeting, structure priors, or CDR-restricted optimization. EasyNano adds these capabilities and
demonstrates their practical effectiveness across six diverse targets.

## Citation

```bibtex
@article{hu2026easynano,
  title   = {EasyNano: rapid epitope-targeted nanobody CDR design via
             differentiable distogram optimization with ESMFold2},
  author  = {Hu, Yue and Cheng, Wanyu and Wang, Junqing and Liu, Yingchao},
  journal = {bioRxiv},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
