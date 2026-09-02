# EasyNano v2

Epitope-targeted nanobody CDR design via ESMFold2 differentiable distogram optimization.

**v2 key change vs v1:** `esmscore` is now **vendored** into this repo as a
top-level `esmscore/` package, and the ESMFold2 model code is pulled in
automatically via a public `git` dependency on the Biohub transformers fork.
So `pip install -e .` makes `import esmscore` **and** `transformers.models.esmfold2`
work out of the box — no separate install, no `sys.path` hack, no hardcoded
machine paths, no manual transformers fork checkout.

> **Non-official.** EasyNano is an independent academic tool from YueHuLab.
> It is **not** affiliated with, endorsed by, or an official product of
> EvolutionaryScale / Meta / the original ESM team. "ESMFold2", "ESMC" and
> related marks belong to their respective owners.

---

## Install (one command)

```bash
git clone https://github.com/YueHuLab/easynano_v2.git
cd easynano_v2
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

This installs **both** the `easynano` and the vendored `esmscore` package,
plus `torch`, `biotite`, `abnumber`, and — importantly — the **Biohub fork
of transformers** (pinned to the same commit EvolutionaryScale's own `esm`
package uses), which provides `transformers.models.esmfold2`
(`ESMFold2Config` / `ESMFold2Model` / `gather_rep_atom_coords` /
`protein_utils`). The standard HuggingFace `transformers` on PyPI does **not**
ship `esmfold2`, so this fork is required and is pulled in automatically.

Verify the install (no model weights needed for these two steps):

```bash
python test_quick.py        # checks easynano imports + target parsing
easynano setup --target ty1_rbd   # exercises the full setup path on test PDBs
```

---

## Model weights (required to actually run `design`)

`esmscore`/`easynano` are just code — the ESMFold2 / ESMC neural weights are
not bundled (they are large and require EvolutionaryScale access). Download
the three weight directories from **https://evolutionaryscale.ai**:

| Env var | Weight set | Size | Used by |
|---|---|---|---|
| `ESMFOLD2_FAST` | ESMFold2-Fast | 721M | `design` loop |
| `ESMFOLD2_FULL` | ESMFold2      | 1.3B | structure prior + `eval`/`baseline` |
| `ESMC_PATH`     | ESMC-6B       | 6B   | sequence encoder backing ESMFold2 |

Then export them (e.g. in your shell rc):

```bash
export ESMFOLD2_FAST=/path/to/ESMFold2-Fast
export ESMFOLD2_FULL=/path/to/ESMFold2
export ESMC_PATH=/path/to/ESMC-6B
export EASYNANO_DEVICE=mps          # or cuda / cpu
```

Run the preflight to confirm everything resolves before launching a design run:

```bash
easynano check
```

If any weight is missing, `easynano check` (and `easynano design`) will print
a clear message telling you exactly which env var to set — instead of crashing
deep inside `from_pretrained`.

---

## Design a nanobody

```bash
# 1) (optional) inspect a target's parsed epitope / framework
easynano setup --target ty1_rbd

# 2) run the CDR design loop on the ty1 receptor-binding domain target
easynano design --target ty1_rbd --seeds 0 1 2 --steps 60

# 3) evaluate top snapshots with the full 1.3B model
easynano eval --target ty1_rbd --snapshots results/seed0_snapshots.json --n-top 5

# 4) random-CDR baseline for comparison
easynano baseline --target ty1_rbd --n-random 30
```

Or all-in-one:

```bash
easynano run --target ty1_rbd --seeds 0 1 2
```

Available targets ship as test PDBs under `test/` (ty1_rbd, ty2_rbd, ty3_rbd,
rsb, b5 — see `easynano/config.py` `TARGETS` registry). To use your own target,
point `--pdb` / `--chain` / `--epitope-indices` at your PDB.

---

## What was fixed in v2 (install pain points)

| Problem in v1 | v2 fix |
|---|---|
| `pyproject.toml` declared the wrong package name (`esm`), so `pip install .` installed EvolutionaryScale's `esm` and `import easynano` failed with `ModuleNotFoundError` | `pyproject.toml` now declares `name = "easynano"` and `include = ["easynano*", "esmscore*"]` |
| `esmscore` was a local-only package at a hardcoded absolute path (`/Users/huyue/esm-c-fold2`); not on PyPI, not in `esm`, not vendored → users had no way to obtain it | `esmscore/` is now vendored at the repo top level; `pip install -e .` installs it |
| `transformers.models.esmfold2` (the ESMFold2 model code: `ESMFold2Config`/`ESMFold2Model`/`gather_rep_atom_coords`/`protein_utils`) is NOT in standard HuggingFace `transformers` on PyPI — it lives in the public Biohub transformers fork. v1 silently relied on users having this fork pre-installed | v2 declares `transformers @ git+https://github.com/Biohub/transformers.git@3a8956f` as an explicit dependency, so it is pulled in automatically |
| `design.py`/`evaluate.py`/`baseline.py` used `sys.path.insert(0, ESM_REPO)` with `ESM_REPO` defaulting to the author's machine path | removed; `import esmscore` now resolves through the installed package |
| `config.py` hardcoded model weight paths to the author's machine | defaults are now `None`; set via env vars; `check_weights()` gives a clear error |
| No preflight — `easynano design` would crash deep inside `from_pretrained` if weights were missing | `easynano check` validates imports + weights up front |

---

## Repo layout

```
easynano_v2/
├── easynano/            # the package (cli, config, design, evaluate, baseline, setup, loss, analyze)
├── esmscore/            # vendored ESMFold2 scoring/complex/pdb helpers
│   ├── __init__.py
│   ├── score_only.py
│   ├── _complex.py
│   ├── _pdb.py
│   └── _pdb_writer.py
├── test/                # bundled target PDBs (ty1/ty2/ty3/rsb/b5)
├── test_quick.py        # install smoke test (no weights needed)
├── pyproject.toml
└── README.md
```

## Citation

If you use EasyNano, please cite the EasyNano manuscript (arXiv:2606.12772)
and the ESMFold2 / ESMC papers from EvolutionaryScale.

## License

See `LICENSE.md`. The vendored `esmscore/` is included for reproducibility;
respect any upstream licensing for the ESMFold2 model weights (separate
download from EvolutionaryScale).
