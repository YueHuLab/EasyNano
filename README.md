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

EasyNano is the installable Python package `easynano`. There are two tiers of
dependencies, and it is important to know the difference:

- **Tier 1 — package + simple test + `setup` (verified, works everywhere).**
  Needs only `biotite`, `abnumber`, `transformers` (standard PyPI), and a few
  plotting libs. **No ESMFold2 weights, no GPU, and no `esmscore`.** This is the
  part that was broken before: the repo's previous `pyproject.toml` actually
  described a *different* package (ESM's `esm` 3.3.0), so `pip install .` installed
  the wrong thing and `import easynano` failed with `ModuleNotFoundError`. That is
  now fixed — `easynano` installs cleanly and the quick test below runs with
  nothing but Tier-1 deps.
- **Tier 2 — full `design` / `eval` / `baseline` pipeline.** These commands
  additionally import **`esmscore`** (an internal module of EvolutionaryScale's
  ESM) and load ESMFold2/ESMC model weights. See "Tier 2 prerequisites" below.

### Step 1 — Clone and install the package (Tier 1)

```bash
# 1. Clone
git clone https://github.com/YueHuLab/EasyNano.git
cd EasyNano

# 2. (Recommended) Python 3.10–3.12 in a fresh venv
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# 3. Install EasyNano + its pip-available dependencies
pip install -e .
```

> **No Biohub fork needed.** `transformers` on PyPI (≥ 4.45) already ships
> `transformers.models.esmfold2`, so you do **not** need the Biohub `transformers`
> fork or any `git+https://` install for it.

### Step 2 — Quick test (Tier 1, no weights, no esmscore)

```bash
easynano setup --target ty1_rbd
python test_quick.py
```

Both should run and print a success line (see **Quick Test** below). If they do,
your install is correct and the package imports fine.

### Step 3 — Tier 2 prerequisites (design / eval / baseline only)

The `design`, `eval`, and `baseline` commands import `esmscore` inside their
function bodies (so the rest of the package — including `setup` and `test_quick.py`
— is unaffected). **`esmscore` is not published to PyPI** and is **not** included
in the current `esm` PyPI package (3.4.0); `esm==3.3.0` (referenced by the old
config) is not on PyPI either. In the authors' environment `esmscore` was available
from a compatible ESM build. To run the full pipeline you must make `esmscore`
importable (e.g. from a compatible ESM source/install) **before** calling those
commands; `import esmscore` should succeed on its own first.

You also need the three model directories (ESMFold2-Fast 721M, ESMFold2 1.3B,
ESMC-6B). Point EasyNano at them with environment variables (no code edits needed):

```bash
export ESMFOLD2_FAST=/path/to/ESMFold2-Fast   # 721M, used in the design loop
export ESMFOLD2_FULL=/path/to/ESMFold2         # 1.3B, structure prior + evaluation
export ESMC_PATH=/path/to/ESMC-6B              # 6B encoder
export EASYNANO_DEVICE=mps                     # mps (Apple Silicon) | cuda | cpu
```

(Defaults assume `/Users/huyue/esm-c-fold2/...`; override with the variables above
on any machine.)

## Quick Test (verify the install — no model weights needed)

The `setup` command only parses a target PDB and detects CDRs, so it needs **no
ESMFold2 weights**. It is the fastest way to confirm EasyNano imported correctly:

```bash
easynano setup --target ty1_rbd
```

Expected output ends with a line like:

```
[OK] target_len=1060 binder_len=118 n_cdr=0 n_epi=21 framework=ty1
```

If you see that line, the package, `biotite`, and `abnumber` are all working.
A self-contained script version is provided at [`test_quick.py`](test_quick.py):

```bash
python test_quick.py
```

To run the actual design loop you must first set the model-weight environment
variables shown above.

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
├── easynano/                   # Installable Python package (pip install -e .)
│   ├── cli.py                  # Command-line entry point: `easynano`
│   ├── config.py               # Model paths, design defaults, target registry
│   ├── setup.py               # Target PDB parsing, epitope/CDR detection
│   ├── design.py              # v9 differentiable CDR design loop
│   ├── evaluate.py            # Full ESMFold2 evaluation
│   ├── loss.py                # Loss functions (epitope, prior, contacts)
│   ├── baseline.py            # Random CDR baselines
│   └── analyze.py             # Developability / supplementary analyses
├── cookbook/tutorials/         # Standalone tutorial scripts (mirror the package)
│   ├── design_target_v9.py     # Main v9 design loop (CA-coordinate prior)
│   ├── design_target.py        # v2 design loop (target-only prior)
│   ├── test_target_pdb.py      # Target setup from arbitrary PDB + epitope
│   ├── loss_functions.py       # Core loss functions
│   ├── eval_all_targets.py     # Batch Full ESMFold2 evaluation
│   ├── eval_target_snapshots.py# Per-target snapshot evaluation
│   ├── random_baseline.py      # Random CDR baseline generation
│   ├── rmsd_validation.py      # Kabsch RMSD cross-validation
│   ├── make_figures.py         # Manuscript figure generation
│   ├── make_cdr_logos.py       # CDR sequence logo generation
│   ├── analyze_b5_losses.py    # Loss function analysis
│   └── run_v9_batch*.sh        # Batch design runners
├── manuscript/                 # Preprint (main.tex/pdf, supplementary, figures)
├── test/                       # Target PDB files (6 targets)
├── test_quick.py               # Simple install smoke test (no weights needed)
├── pyproject.toml             # Package metadata (name = "easynano")
└── LICENSE
```

## Quick Start

> **Prerequisite:** set the model-weight environment variables (see *Installation* above).
> Without them, `design`/`eval` will fail to load ESMFold2.

```bash
# Point EasyNano at your ESMFold2 weights (replace with your paths)
export ESMFOLD2_FAST=/path/to/ESMFold2-Fast
export ESMFOLD2_FULL=/path/to/ESMFold2
export ESMC_PATH=/path/to/ESMC-6B
export EASYNANO_DEVICE=mps          # mps (Apple Silicon) | cuda | cpu

# 1. Setup target (no weights needed — safe to run first)
easynano setup --target kn035_pdl1

# 2. Run v9 design (CA-coordinate prior) — needs weights
easynano design --target kn035_pdl1 --steps 5 --seed 0

# 3. Evaluate top candidates with Full ESMFold2 — needs weights
easynano eval --target kn035_pdl1 \
    --snapshots results/kn035_pdl1/seed0_snapshots.json --n-top 3

# 4. (Optional) random CDR baselines for significance testing — needs weights
easynano baseline --target kn035_pdl1 --n-random 30
```

All commands accept `--target <name>` from the built-in registry (see *Targets Evaluated*)
or explicit `--pdb/--chain/--epitope-indices/--framework` overrides. Run
`easynano <command> --help` for every flag.

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
