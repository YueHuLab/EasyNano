"""Unified command-line interface for EasyNano.

Usage:
    easynano setup   --target ty1_rbd
    easynano design  --target ty1_rbd --seeds 0 1 2
    easynano eval    --target ty1_rbd --snapshots results/seed0_snapshots.json
    easynano baseline --target ty1_rbd --n-random 30
    easynano analyze --target ty1_rbd --sequence <CDR_SEQ>
    easynano run     --target ty1_rbd --seeds 0 1 2  # all-in-one
"""

from __future__ import annotations

import sys
import argparse

from .config import TARGETS


def _get_target(name: str) -> dict:
    if name not in TARGETS:
        print(f"Unknown target '{name}'. Available: {list(TARGETS)}")
        sys.exit(1)
    return TARGETS[name]


def _parse_epitope(cfg: dict) -> list[int]:
    if isinstance(cfg["epitope"], str):
        return [int(x) for x in cfg["epitope"].split(",") if x.strip()]
    return list(cfg["epitope"])


def cmd_setup(args):
    """Setup target and print diagnostics."""
    from .setup import setup_target_design
    cfg = _get_target(args.target)
    epi = _parse_epitope(cfg) if not args.epitope_indices else \
        [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    setup = setup_target_design(
        pdb_path=args.pdb or cfg["pdb"],
        target_chain=args.chain or cfg["chain"],
        epitope_indices=epi,
        framework=args.framework or cfg["framework"],
    )
    print(f"\n[OK] target_len={len(setup['target_sequence'])} "
          f"binder_len={len(setup['binder_template'])} "
          f"n_cdr={len(setup['cdr_indices'])} "
          f"n_epi={len(setup['epitope_token_indices'])} "
          f"framework={setup['framework']}")


def cmd_design(args):
    """Run CDR design."""
    from .design import run_design
    cfg = _get_target(args.target)
    epi = _parse_epitope(cfg) if not args.epitope_indices else \
        [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    pdb = args.pdb or cfg["pdb"]
    chain = args.chain or cfg["chain"]
    framework = args.framework or cfg["framework"]
    out_dir = args.out_dir or f"results/{args.target}"

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        run_design(
            pdb_path=pdb, target_chain=chain,
            epitope_indices=epi, framework=framework,
            seed=seed,
            steps=args.steps, lr=args.lr,
            wt_logit=args.wt_logit,
            w_epitope=args.w_epitope, w_intra=args.w_intra,
            w_inter=args.w_inter, w_glob=args.w_glob,
            w_prior=args.w_prior, w_aa_freq=args.w_aa_freq,
            sample_steps=args.sample_steps, n_loops=args.n_loops,
            skip_prior=args.skip_prior,
            pin_cys_in_cdr=not args.allow_cdr_cys,
            out_dir=out_dir,
        )


def cmd_eval(args):
    """Evaluate design snapshots with Full ESMFold2."""
    from .evaluate import evaluate_snapshots
    cfg = _get_target(args.target)
    epi = _parse_epitope(cfg) if not args.epitope_indices else \
        [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    evaluate_snapshots(
        pdb_path=args.pdb or cfg["pdb"],
        target_chain=args.chain or cfg["chain"],
        epitope_indices=epi,
        framework=args.framework or cfg["framework"],
        snapshot_path=args.snapshots,
        n_top=args.n_top,
        out=args.out,
        include_wt=not args.no_wt,
    )


def cmd_baseline(args):
    """Run random CDR baseline."""
    from .baseline import run_baseline
    cfg = _get_target(args.target)
    epi = _parse_epitope(cfg) if not args.epitope_indices else \
        [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    run_baseline(
        pdb_path=args.pdb or cfg["pdb"],
        target_chain=args.chain or cfg["chain"],
        epitope_indices=epi,
        framework=args.framework or cfg["framework"],
        n_random=args.n_random,
        seed=args.seed,
        out=args.out,
    )


def cmd_analyze(args):
    """Run developability and supplementary analyses."""
    from .analyze import developability_report, print_developability_report

    if args.developability:
        seq = args.sequence
        if not seq:
            # Use framework WT
            cfg = _get_target(args.target)
            from .config import INIT_FRAMEWORKS
            seq = INIT_FRAMEWORKS.get(cfg["framework"], "")
            if not seq:
                print("No sequence provided. Use --sequence or specify a target with a framework.")
                return
        report = developability_report(seq, label=args.label or "sequence")
        print_developability_report(report)
        if args.out:
            import json
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)


def cmd_run(args):
    """All-in-one: setup → design → evaluate → baseline."""
    print(f"\n{'='*60}")
    print(f"  EasyNano full pipeline: {args.target}")
    print(f"{'='*60}")

    # 1. Setup
    print("\n--- Stage 1: Setup ---")
    cmd_setup(args)

    # 2. Design
    print("\n--- Stage 2: Design ---")
    cmd_design(args)

    # 3. Evaluate (pick best seed)
    print("\n--- Stage 3: Evaluate ---")
    import glob
    out_dir = args.out_dir or f"results/{args.target}"
    for seed in args.seeds:
        snap = f"{out_dir}/seed{seed}_snapshots.json"
        import os
        if os.path.exists(snap):
            args.snapshots = snap
            args.out = f"{out_dir}/eval_seed{seed}.json"
            cmd_eval(args)

    # 4. Baseline
    print("\n--- Stage 4: Random Baseline ---")
    cmd_baseline(args)


def main():
    parser = argparse.ArgumentParser(
        description="EasyNano: rapid epitope-targeted nanobody CDR design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easynano setup --target ty1_rbd
  easynano design --target ty1_rbd --seeds 0 1 2 --out-dir results/ty1
  easynano eval --target ty1_rbd --snapshots results/ty1/seed0_snapshots.json
  easynano baseline --target ty1_rbd --n-random 30
  easynano run --target ty1_rbd --seeds 0 1 2
  easynano analyze --developability --sequence GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIY
        """,
    )
    parser.add_argument("--version", action="version",
                        version=f"EasyNano {__import__('easynano').__version__}")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Common target args
    def add_target_args(p):
        p.add_argument("--target", help="Target name from registry (or use --pdb/--chain/--epitope-indices)")
        p.add_argument("--pdb", help="Path to target PDB (overrides registry)")
        p.add_argument("--chain", help="Target chain ID (overrides registry)")
        p.add_argument("--epitope-indices", help="Comma-separated epitope residue indices")
        p.add_argument("--framework", help="Framework name (overrides registry)")
        p.add_argument("--out-dir", default="results", help="Output directory")

    # setup
    p_setup = sub.add_parser("setup", help="Setup target and print diagnostics")
    add_target_args(p_setup)

    # design
    p_design = sub.add_parser("design", help="Run CDR design loop")
    add_target_args(p_design)
    p_design.add_argument("--seeds", type=int, nargs="+", default=[0],
                          help="Random seeds (use ≥3)")
    p_design.add_argument("--steps", type=int, default=60)
    p_design.add_argument("--lr", type=float, default=0.05)
    p_design.add_argument("--wt-logit", type=float, default=2.0,
                          help="WT logit bias (2.0=sweet spot, 5.0=locked)")
    p_design.add_argument("--w-epitope", type=float, default=0.2)
    p_design.add_argument("--w-intra", type=float, default=0.5)
    p_design.add_argument("--w-inter", type=float, default=0.5)
    p_design.add_argument("--w-glob", type=float, default=0.2)
    p_design.add_argument("--w-prior", type=float, default=0.05,
                          help="Structure prior weight (0.05=sweet spot)")
    p_design.add_argument("--w-aa-freq", type=float, default=0.01)
    p_design.add_argument("--sample-steps", type=int, default=5)
    p_design.add_argument("--n-loops", type=int, default=1)
    p_design.add_argument("--skip-prior", action="store_true")
    p_design.add_argument("--allow-cdr-cys", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate designs with Full ESMFold2")
    add_target_args(p_eval)
    p_eval.add_argument("--snapshots", required=True, help="Path to snapshot JSON")
    p_eval.add_argument("--n-top", type=int, default=5)
    p_eval.add_argument("--out", help="Output JSON path")
    p_eval.add_argument("--no-wt", action="store_true")

    # baseline
    p_base = sub.add_parser("baseline", help="Run random CDR baseline")
    add_target_args(p_base)
    p_base.add_argument("--n-random", type=int, default=30)
    p_base.add_argument("--seed", type=int, default=42)
    p_base.add_argument("--out", help="Output JSON path")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Supplementary analyses")
    p_analyze.add_argument("--target", help="Target name from registry")
    p_analyze.add_argument("--developability", action="store_true")
    p_analyze.add_argument("--sequence", help="Binder/CDR sequence to analyze")
    p_analyze.add_argument("--label", default="", help="Label for the report")
    p_analyze.add_argument("--out", help="Output JSON path")

    # run (all-in-one)
    p_run = sub.add_parser("run", help="Run full pipeline (setup→design→eval→baseline)")
    add_target_args(p_run)
    p_run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_run.add_argument("--steps", type=int, default=60)
    p_run.add_argument("--lr", type=float, default=0.05)
    p_run.add_argument("--wt-logit", type=float, default=2.0)
    p_run.add_argument("--w-epitope", type=float, default=0.2)
    p_run.add_argument("--w-prior", type=float, default=0.05)
    p_run.add_argument("--n-random", type=int, default=30)
    p_run.add_argument("--allow-cdr-cys", action="store_true")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "design":
        cmd_design(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
