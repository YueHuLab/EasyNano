"""Finalize the quick-validation report by reading the run + eval + RMSD outputs.

Reads:
  - /tmp/quick_val/runs/<tag>_seed<N>.json   (design-loop snapshots)
  - /tmp/quick_val/evals/_all_targets_summary.json  (Full-ESMFold2 per-step)
  - /tmp/quick_val/rmsd_summary.json         (Kabsch cross-validation)
  - /tmp/quick_val/evals/<tag>_seed<N>_eval.json  (per-seed Full eval, for WT init)

Writes:
  - /tmp/quick_val/finalized_report.md  (a copy of the report with tables filled in)
  - Prints the filled-in tables to stdout
"""
from __future__ import annotations
import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np

REPORT_PATH = Path(__file__).resolve().parent / "quick_val_report.md"

TARGETS = [
    ("PDL1_5JDS_KN035",  "PD-L1 (5JDS)", "KN035"),
    ("RBD_6WAQ_VHH72",   "RBD (6WAQ)",   "VHH-72"),
    ("TNFA_5M2J_ANTITNF","TNFα (5M2J)",  "anti-TNF"),
    ("RBD_6ZXN_TY1",     "RBD (6ZXN)",   "Ty1"),
    ("TNFA_5M2M_VHH3",   "TNFα (5M2M)",  "VHH3"),
]


def collect_design_loop(runs_dir: str) -> dict:
    """Read per-seed design-loop snapshots; return per-target summary."""
    by_tag: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(f"{runs_dir}/*.json")):
        stem = Path(f).stem
        tag = "_seed".join(stem.split("_seed")[:-1])
        with open(f) as fh:
            d = json.load(fh)
        by_tag.setdefault(tag, []).append(d)

    out = {}
    for tag, runs in by_tag.items():
        # Per-seed best snapshot (highest step with the lowest inter loss)
        L_epi_init = []
        L_epi_best = []
        cdr_to_epi_init = []
        cdr_to_epi_best = []
        n_mut = []
        basin_step = []
        for d in runs:
            snaps = d["snapshots"]
            if not snaps:
                continue
            init = snaps[0]
            best = min(snaps[1:], key=lambda s: s.get("inter", 1e9)) if len(snaps) > 1 else init
            L_epi_init.append(init.get("epi", 0.0))
            L_epi_best.append(best.get("epi", 0.0))
            cdr_to_epi_init.append(init.get("cdr_to_epi_min", 0.0))
            cdr_to_epi_best.append(best.get("cdr_to_epi_min", 0.0))
            # n_mutated: count positions where design changed WT
            wt = d.get("init_cdr", "")
            best_cdr = best.get("cdr_seq", "")
            n_mut.append(sum(1 for a, b in zip(wt, best_cdr) if a != b))
            # steps to basin: first step with L_epi < 1.0 (a generous basin threshold)
            bs = next((s["step"] for s in snaps if s.get("epi", 1e9) < 1.0), None)
            basin_step.append(bs if bs is not None else 100)
        out[tag] = {
            "n_seeds": len(runs),
            "L_epi_init": L_epi_init,
            "L_epi_best": L_epi_best,
            "cdr_to_epi_init": cdr_to_epi_init,
            "cdr_to_epi_best": cdr_to_epi_best,
            "n_mutated": n_mut,
            "basin_step": basin_step,
        }
    return out


def collect_full_eval(evals_dir: str) -> dict:
    """Read _all_targets_summary.json and individual eval files for WT init."""
    with open(Path(evals_dir) / "_all_targets_summary.json") as f:
        all_summaries = json.load(f)
    out = {}
    for tag, payload in all_summaries.items():
        summary = payload["summary"]
        # Find step 0 (WT init) — it's a separate "step" key
        wt = next((s for s in summary if s["step"] == -1), None)
        # Best design: max iptm_median across steps
        design_steps = [s for s in summary if s["step"] >= 0]
        best = max(design_steps, key=lambda s: s["iptm_median"] or -1, default=None)
        out[tag] = {
            "wt": wt,
            "best": best,
            "n_folded_steps": sum(s["n"] for s in design_steps),
        }
    return out


def collect_rmsd(rmsd_path: str) -> dict:
    with open(rmsd_path) as f:
        return json.load(f)


def fmt(vals, fmtstr="{:.2f}"):
    if not vals:
        return "—"
    med = np.median(vals)
    lo, hi = min(vals), max(vals)
    return f"{fmtstr.format(med)} [{fmtstr.format(lo)}, {fmtstr.format(hi)}]"


def render_table(rows: list, headers: list) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/tmp/quick_val/runs")
    p.add_argument("--evals-dir", default="/tmp/quick_val/evals")
    p.add_argument("--rmsd-path", default="/tmp/quick_val/rmsd_summary.json")
    p.add_argument("--out", default="/tmp/quick_val/finalized_report.md")
    args = p.parse_args()

    if not Path(args.rmsd_path).exists():
        print(f"[WARN] {args.rmsd_path} missing — RMSD rows will be _TBD_")
        rmsd = {}
    else:
        rmsd = collect_rmsd(args.rmsd_path)
    design = collect_design_loop(args.runs_dir)
    full = collect_full_eval(args.evals_dir)

    # Table 1: design loop
    table1 = []
    for tag, label, fw in TARGETS:
        d = design.get(tag, {})
        if not d:
            table1.append([label, fw, "—", "—", "—", "—", "—", "—"])
            continue
        table1.append([
            label, fw,
            fmt(d["L_epi_init"], "{:.2f}"),
            fmt(d["L_epi_best"], "{:.2f}"),
            fmt(d["cdr_to_epi_init"], "{:.2f}"),
            fmt(d["cdr_to_epi_best"], "{:.2f}"),
            f"{int(np.median(d['n_mutated']))} [{min(d['n_mutated'])}, {max(d['n_mutated'])}]",
            f"{int(np.median(d['basin_step']))}",
        ])

    # Table 2: Full ESMFold2
    table2 = []
    for tag, label, _ in TARGETS:
        e = full.get(tag, {})
        wt = e.get("wt") or {}
        best = e.get("best") or {}
        if not wt or not best:
            table2.append([label, "—", "—", "—", "—", "—", "—"])
            continue
        wt_iptm = wt["iptm_median"]
        wt_ptm = wt["ptm_median"]
        best_iptm = best["iptm_median"]
        best_ptm = best["ptm_median"]
        delta = (best_iptm - wt_iptm) if (best_iptm is not None and wt_iptm is not None) else None
        table2.append([
            label,
            f"{wt_iptm:.3f}" if wt_iptm is not None else "—",
            f"{wt_ptm:.3f}" if wt_ptm is not None else "—",
            f"{best_iptm:.3f}" if best_iptm is not None else "—",
            f"{best_ptm:.3f}" if best_ptm is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
            f"{e['n_folded_steps']}",
        ])

    # Table 3: RMSD
    table3 = []
    for tag, label, _ in TARGETS:
        r = rmsd.get(tag, {})
        if not r or "pose_comparison" not in r:
            table3.append([label, "—", "—", "—", "—", "—"])
            continue
        pc = r["pose_comparison"]
        init = pc.get("WT_init", {})
        des = pc.get("best_design", {})
        if not init or not des:
            table3.append([label, "—", "—", "—", "—", "—"])
            continue
        init_rmsd = init.get("binder_rmsd_A", 0)
        des_rmsd = des.get("binder_rmsd_A", 0)
        delta = des_rmsd - init_rmsd
        # `delta` ≈ 0 means design preserves framework's predicted pose
        # (large absolute vs-crystal is expected — see [[esmfold2_pose_basin]])
        table3.append([
            label,
            f"{init_rmsd:.2f}",
            f"{des_rmsd:.2f}",
            f"{delta:+.2f}",
            f"{des.get('target_rmsd_A', 0):.2f}",
            f"{des.get('n_pred_interface_contacts', 0)} / {init.get('n_real_interface_contacts', 0)}",
        ])

    # Build findings
    findings = []
    pos_count = 0
    pose_ok = 0
    for tag, label, _ in TARGETS:
        e = full.get(tag, {})
        wt = (e.get("wt") or {}).get("iptm_median")
        best = (e.get("best") or {}).get("iptm_median")
        if wt is not None and best is not None and best > wt:
            pos_count += 1
        r = rmsd.get(tag, {})
        bd = r.get("pose_comparison", {}).get("best_design", {})
        if bd and bd.get("binder_rmsd_A", 99) < 10.0:
            pose_ok += 1
    findings.append(f"1. **Does v2 work for arbitrary new targets?** "
                    f"{pos_count}/3 targets show design_iptm > WT_iptm.")
    findings.append(f"2. **Are the improvements real or noise?** "
                    f"See per-target std in Full ESMFold2 table; "
                    f"B5 lesson is single-seed iptm is noisy — multi-seed medians are more reliable.")
    findings.append(f"3. **Is the framework pose preserved (|design_RMSD − WT_RMSD| ≤ ~5 Å)?** "
                    f"Per [[esmfold2_pose_basin]], the relevant metric is the *delta* "
                    f"between design and WT_init binder-RMSD vs the crystal VHH, not the "
                    f"absolute value (the framework's predicted pose can differ from the "
                    f"crystal pose by 10s of Å even for the WT). A small delta means the "
                    f"design preserved the framework's predicted pose.")
    findings.append(f"4. **What are the failure modes, if any?** "
                    f"Inspect CDR→epi (Å) at the end of design: large values mean "
                    f"the design didn't reach the epitope.")

    report = REPORT_PATH.read_text()

    # Replace tables
    def replace_table(text, header_marker, new_body):
        # Find table block — header line + sep + body lines until blank or next section
        lines = text.splitlines()
        out_lines = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if header_marker in ln and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
                # skip header + sep
                out_lines.append(ln)
                out_lines.append(lines[i + 1])
                # find body lines (consecutive | lines)
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                # insert new body
                out_lines.append(new_body)
                i = j
            else:
                out_lines.append(ln)
                i += 1
        return "\n".join(out_lines)

    t1_body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in table1)
    t2_body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in table2)
    t3_body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in table3)

    report = replace_table(report, "Target | Framework | L_epi step 0", t1_body)
    report = replace_table(report, "Target | WT init iptm", t2_body)
    report = replace_table(report, "Target | init (WT) binder-RMSD", t3_body)

    # Replace findings block
    if "## Findings" in report:
        before, _, after = report.partition("## Findings")
        # find next "## " section
        end = after.find("\n## ")
        if end == -1:
            after_section = ""
        else:
            after_section = after[end:]
            after = after[:end]
        findings_block = "## Findings\n\n" + "\n".join(findings) + "\n\n"
        report = before + findings_block + after_section.lstrip("\n")

    Path(args.out).write_text(report)
    print(f"Wrote {args.out}\n")
    print("=" * 80)
    print("DESIGN LOOP:")
    print(render_table(table1, ["Target", "Framework", "L_epi init", "L_epi best",
                                "CDR→epi init (Å)", "CDR→epi best (Å)",
                                "n_mutated", "#steps to basin"]))
    print()
    print("FULL ESMFold2:")
    print(render_table(table2, ["Target", "WT init iptm", "WT init pTM",
                                "Best design iptm", "Best design pTM",
                                "iptm Δ", "n_folded"]))
    print()
    print("KABSCH RMSD:")
    print(render_table(table3, ["Target", "WT_init binder-RMSD (Å)",
                                "best_design binder-RMSD (Å)", "Δ (des−init) (Å)",
                                "target-RMSD (Å)", "n_interface (pred/real)"]))
    print()
    print("FINDINGS:")
    for f in findings:
        print("  " + f)


if __name__ == "__main__":
    main()
