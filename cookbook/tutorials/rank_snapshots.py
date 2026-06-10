"""Inspect design snapshot JSON and rank by design-time metrics.

Reads a snapshot file and prints the snapshots in order of multiple metrics
(inter loss, epitope loss, CDR→epi distance, etc.) so we can pick the best
ones to re-evaluate with the FULL model at high quality.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, default="/tmp/b5_design_v2_snaps.json")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    with open(args.snapshots) as f:
        data = json.load(f)
    snapshots = data["snapshots"]
    init_cdr = data.get("init_cdr", "?")

    print(f"Snapshot file: {args.snapshots}")
    print(f"  init_cdr: {init_cdr}")
    print(f"  {len(snapshots)} snapshots\n")

    # Compute combined score: -epi + -CDR→epi (lower is better)
    # so high score = better
    for s in snapshots:
        s["score"] = -s["epi"] - s["cdr_to_epi_min"]

    # Print all snapshots in step order
    print(f"  {'step':>4}  {'inter':>7}  {'epi':>7}  {'CDR→epi':>8}  "
          f"{'inter_min':>10}  {'pTM':>5}  {'ipTM':>5}  CDR_seq")
    for s in snapshots:
        ptm_s = f"{s.get('ptm', 0):.3f}" if s.get('ptm') else "  N/A"
        iptm_s = f"{s.get('iptm', 0):.3f}" if s.get('iptm') else "  N/A"
        print(f"  {s['step']:>4}  {s['inter']:>7.3f}  {s['epi']:>7.3f}  "
              f"{s['cdr_to_epi_min']:>8.2f}  {s['inter_min']:>10.2f}  "
              f"{ptm_s:>5}  {iptm_s:>5}  {s['cdr_seq']}")

    # Rankings
    print(f"\n=== Top {args.top_k} by inter loss ===")
    for s in sorted(snapshots, key=lambda x: x["inter"])[:args.top_k]:
        print(f"  step={s['step']:>3}  inter={s['inter']:.4f}  epi={s['epi']:.2f}  "
              f"CDR→epi={s['cdr_to_epi_min']:.2f}Å  {s['cdr_seq']}")

    print(f"\n=== Top {args.top_k} by epitope loss ===")
    for s in sorted(snapshots, key=lambda x: x["epi"])[:args.top_k]:
        print(f"  step={s['step']:>3}  epi={s['epi']:.2f}  CDR→epi={s['cdr_to_epi_min']:.2f}Å  "
              f"inter={s['inter']:.4f}  {s['cdr_seq']}")

    print(f"\n=== Top {args.top_k} by CDR→epi distance ===")
    for s in sorted(snapshots, key=lambda x: x["cdr_to_epi_min"])[:args.top_k]:
        print(f"  step={s['step']:>3}  CDR→epi={s['cdr_to_epi_min']:.2f}Å  "
              f"inter={s['inter']:.4f}  epi={s['epi']:.2f}  {s['cdr_seq']}")


if __name__ == "__main__":
    main()
