"""Generate preprint figures from v9 design results + Full ESMFold2 eval + random baselines.

Output: /tmp/v9_designs/figures/
  - fig1_main_results.pdf/png — WT vs Design ipTM bar chart + random baseline overlay
  - fig2_trajectories.pdf/png — cdr→epi vs step for best seed per target
  - fig3_random_dist.pdf/png — random baseline distributions with design marked
  - table1_summary.tex — LaTeX summary table
"""
from __future__ import annotations
import json, os, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

OUT = Path("/tmp/v9_designs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load all data ----
def load_snaps(tag, seed):
    p = f"/tmp/v9_designs/{tag}_seed{seed}_snaps.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d

def load_random(tag):
    p = f"/tmp/v9_designs/{tag}_random.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))

summary = json.load(open("/tmp/v9_designs/SUMMARY_FULL.json"))

TARGETS = [
    ("RBD_6ZXN_TY1",      "Ty1 / RBD",          "#2196F3"),
    ("PDL1_5JDS",         "KN035 / PD-L1",      "#4CAF50"),
    ("TNFA_5M2M",         "VHH3 / TNFα",        "#FF9800"),
    ("RBD_6WAQ_VHH72",    "VHH72 / RBD",        "#9C27B0"),
    ("TNFA_5M2J",         "anti-TNF / TNFα",    "#F44336"),
]

# ---- Style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================
# FIGURE 1: Main results — WT vs Design ipTM bar chart
# ============================================================
def fig1_main_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    labels = []
    wt_vals, des_vals = [], []
    rand_meds, rand_stds = [], []
    colors = []

    for tag, label, color in TARGETS:
        ev = summary.get(tag, {})
        bl = load_random(tag)
        wt = ev.get("wt_iptm", 0)
        best = ev.get("best_design_iptm") or 0
        labels.append(label)
        wt_vals.append(wt)
        des_vals.append(best)
        colors.append(color)
        if bl:
            rand_meds.append(bl["random_iptm_median"])
            rand_stds.append(bl["random_iptm_std"])
        else:
            rand_meds.append(0)
            rand_stds.append(0)

    x = np.arange(len(labels))
    w = 0.3

    bars_wt = ax1.bar(x - w/2, wt_vals, w, label="WT (initial)", color="#BDBDBD", edgecolor="#757575", linewidth=0.5)
    bars_des = ax1.bar(x + w/2, des_vals, w, label="Design (best)", color=colors, edgecolor="#333333", linewidth=1.2)

    # Add random baseline as error bars on a separate axis marker
    for i, (rm, rs) in enumerate(zip(rand_meds, rand_stds)):
        if rm > 0:
            ax1.axhline(y=rm, xmin=(i-0.4)/len(x), xmax=(i+0.4)/len(x),
                       color="#FF5722", linewidth=1.5, linestyle="--", alpha=0.6)
            ax1.text(i, rm + 0.02, f"random\n{rm:.2f}±{rs:.2f}",
                    ha="center", fontsize=7, color="#FF5722", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("ipTM", fontsize=12)
    ax1.set_title("WT vs Design ipTM (Full ESMFold2)", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_ylim(0, 0.95)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate delta on bars
    for i, (wt, des) in enumerate(zip(wt_vals, des_vals)):
        delta = des - wt
        y = max(wt, des) + 0.03
        sign = "+" if delta >= 0 else ""
        color_delta = "#2E7D32" if delta > 0.05 else ("#C62828" if delta < -0.02 else "#757575")
        ax1.annotate(f"{sign}{delta:.3f}", (x[i] + w/2, y),
                    ha="center", fontsize=8, fontweight="bold", color=color_delta)

    # Right panel: delta summary
    ax2.axhline(y=0, color="black", linewidth=0.8)
    deltas = np.array(des_vals) - np.array(wt_vals)
    bar_colors = ["#2E7D32" if d > 0 else "#C62828" for d in deltas]
    ax2.barh(labels, deltas, color=bar_colors, edgecolor="#333333", linewidth=0.5, height=0.6)
    for i, d in enumerate(deltas):
        ax2.text(d + (0.03 if d >= 0 else -0.03), i, f"{d:+.3f}",
                va="center", fontsize=8, fontweight="bold",
                ha="left" if d >= 0 else "right")
    ax2.set_title("Δ ipTM", fontweight="bold")
    ax2.set_xlabel("Δ ipTM (design − WT)")
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("Epitope-Targeted Nanobody CDR Design with ESMFold2",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "fig1_main_results.pdf")
    fig.savefig(OUT / "fig1_main_results.png")
    plt.close()
    print("Fig1 saved.")


# ============================================================
# FIGURE 2: Design trajectories — cdr→epi vs step
# ============================================================
def fig2_trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (tag, label, color) in enumerate(TARGETS):
        ax = axes[idx]
        ev = summary.get(tag, {})

        # Plot all 3 seeds
        for seed in [0, 1, 2]:
            d = load_snaps(tag, seed)
            if d is None:
                continue
            snaps = d["snapshots"]
            steps = [s["step"] for s in snaps]
            epi = [s["cdr_to_epi_min"] for s in snaps]
            epi_init = epi[0] if epi else 0
            epi_best = min(epi)
            alpha = 0.9 if epi_best == min(epi) else 0.4
            lw = 2.0 if epi_best == min(epi) else 0.8
            ax.plot(steps, epi, color=color, alpha=alpha, linewidth=lw,
                   marker="o" if epi_best == min(epi) else "",
                   markersize=4, label=f"seed {seed}" if epi_best == min(epi) else "")

        # WT cdr→epi line
        wt_epi = ev.get("wt_cdr_to_epi", None)
        if wt_epi is not None:
            ax.axhline(y=wt_epi, color="gray", linestyle=":", linewidth=1,
                      label=f"WT ({wt_epi:.1f}Å)")

        ax.set_xlabel("Step")
        ax.set_ylabel("CDR→epitope min (Å)")
        ax.set_title(f"{label}  (WT ipTM={ev.get('wt_iptm',0):.3f})", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, linestyle="--")

    # Remove extra subplot (5 targets, 6 slots)
    axes[5].set_visible(False)

    fig.suptitle("CDR→Epitope Distance During Design Optimization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig2_trajectories.pdf")
    fig.savefig(OUT / "fig2_trajectories.png")
    plt.close()
    print("Fig2 saved.")


# ============================================================
# FIGURE 3: Random baseline distributions
# ============================================================
def fig3_random_dist():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (tag, label, color) in enumerate(TARGETS):
        ax = axes[idx]
        ev = summary.get(tag, {})
        bl = load_random(tag)
        if bl is None:
            continue

        rand_iptms = [r["iptm"] for r in bl["results"] if r["name"].startswith("random")]
        wt_iptm = bl["wt_iptm"]
        design_iptm = ev.get("best_design_iptm") or 0

        # Histogram
        ax.hist(rand_iptms, bins=15, color="#E0E0E0", edgecolor="#9E9E9E", density=True, alpha=0.8)
        ax.axvline(wt_iptm, color="gray", linewidth=2, linestyle="--", label=f"WT ({wt_iptm:.3f})")
        ax.axvline(design_iptm, color=color, linewidth=2.5, label=f"Design ({design_iptm:.3f})")

        # P90 line
        p90 = np.percentile(rand_iptms, 90)
        ax.axvline(p90, color="#FF9800", linewidth=1, linestyle=":", alpha=0.7,
                  label=f"P90 ({p90:.3f})")

        ax.set_xlabel("ipTM")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

        # Add z-score annotation
        rand_mean = np.mean(rand_iptms)
        rand_std = np.std(rand_iptms)
        z = (design_iptm - rand_mean) / max(rand_std, 1e-8)
        ax.text(0.95, 0.95, f"Design: {z:.1f}σ", transform=ax.transAxes,
               ha="right", va="top", fontsize=9, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[5].set_visible(False)
    fig.suptitle("Random CDR Baseline Distributions (Full ESMFold2 ipTM, n=30 each)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig3_random_dist.pdf")
    fig.savefig(OUT / "fig3_random_dist.png")
    plt.close()
    print("Fig3 saved.")


# ============================================================
# FIGURE 4: Combined per-target summary (trajectory + dist side by side)
# ============================================================
def fig4_per_target():
    for tag, label, color in TARGETS:
        ev = summary.get(tag, {})
        bl = load_random(tag)
        if bl is None:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: trajectory
        for seed in [0, 1, 2]:
            d = load_snaps(tag, seed)
            if d is None:
                continue
            snaps = d["snapshots"]
            steps = [s["step"] for s in snaps]
            epi = [s["cdr_to_epi_min"] for s in snaps]
            ax1.plot(steps, epi, color=color, alpha=0.5 + 0.25 * (seed == 0), linewidth=1.5)
        wt_epi = ev.get("wt_cdr_to_epi", None)
        if wt_epi is not None:
            ax1.axhline(y=wt_epi, color="gray", linestyle=":", linewidth=1.5, label=f"WT={wt_epi:.1f}Å")
        ax1.set_xlabel("Step"); ax1.set_ylabel("CDR→epitope (Å)")
        ax1.set_title("Design Trajectory"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        # Right: random distribution
        rand_iptms = [r["iptm"] for r in bl["results"] if r["name"].startswith("random")]
        ax2.hist(rand_iptms, bins=15, color="#E0E0E0", edgecolor="#9E9E9E", alpha=0.8)
        ax2.axvline(bl["wt_iptm"], color="gray", linewidth=2, linestyle="--", label=f"WT={bl['wt_iptm']:.3f}")
        des_iptm = ev.get("best_design_iptm") or 0
        ax2.axvline(des_iptm, color=color, linewidth=2.5, label=f"Design={des_iptm:.3f}")
        ax2.set_xlabel("ipTM"); ax2.set_title("Random Baseline"); ax2.legend(fontsize=8)

        fig.suptitle(f"{label}  |  WT ipTM={ev.get('wt_iptm',0):.3f} → Design ipTM={des_iptm:.3f}  (Δ={des_iptm-ev.get('wt_iptm',0):+.3f})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe_tag = tag.replace("/", "_")
        fig.savefig(OUT / f"fig4_{safe_tag}.pdf")
        fig.savefig(OUT / f"fig4_{safe_tag}.png")
        plt.close()
    print("Fig4 (per-target) saved.")


# ============================================================
# TABLE 1: LaTeX summary table
# ============================================================
def table1_latex():
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full ESMFold2 evaluation of epitope-targeted nanobody CDR design.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Target (PDB) & Framework & WT ipTM & Design ipTM & $\Delta$ ipTM & Random $\mu\pm\sigma$ & $z\sigma$ \\")
    lines.append(r"\midrule")

    for tag, label, color in TARGETS:
        ev = summary.get(tag, {})
        bl = load_random(tag)
        wt = ev.get("wt_iptm", 0)
        des = ev.get("best_design_iptm") or 0
        delta = des - wt

        # Framework name
        fw_map = {
            "RBD_6ZXN_TY1": "Ty1", "PDL1_5JDS": "KN035",
            "TNFA_5M2M": "VHH3", "RBD_6WAQ_VHH72": "VHH72",
            "TNFA_5M2J": "anti-TNF",
        }

        if bl:
            rm = bl["random_iptm_median"]
            rs = bl["random_iptm_std"]
            rand_mean = bl["random_iptm_mean"]
            z = (des - rand_mean) / max(rs, 1e-8)
            rand_str = f"{rm:.3f}$\\pm${rs:.3f}"
            z_str = f"{z:.1f}"
        else:
            rand_str = "—"
            z_str = "—"

        bold_start = "\\textbf{" if delta > 0.1 else ""
        bold_end = "}" if delta > 0.1 else ""

        pdb_map = {"RBD_6ZXN_TY1": "6ZXN", "PDL1_5JDS": "5JDS",
                   "TNFA_5M2M": "5M2M", "RBD_6WAQ_VHH72": "6WAQ", "TNFA_5M2J": "5M2J"}
        lines.append(f"{label} ({pdb_map[tag]}) & {fw_map[tag]} & "
                     f"{bold_start}{wt:.3f}{bold_end} & "
                     f"{bold_start}{des:.3f}{bold_end} & "
                     f"{bold_start}{delta:+.3f}{bold_end} & "
                     f"{rand_str} & {z_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    (OUT / "table1_summary.tex").write_text(latex)
    print("Table1 (LaTeX) saved.")

    # Also write a markdown version
    md = []
    md.append("| Target | Framework | WT ipTM | Design ipTM | Δ ipTM | Random (μ±σ) | zσ |")
    md.append("|--------|-----------|---------|-------------|--------|---------------|-----|")
    for tag, label, color in TARGETS:
        ev = summary.get(tag, {})
        bl = load_random(tag)
        wt = ev.get("wt_iptm", 0)
        des = ev.get("best_design_iptm") or 0
        delta = des - wt
        fw_map = {"RBD_6ZXN_TY1": "Ty1", "PDL1_5JDS": "KN035",
                  "TNFA_5M2M": "VHH3", "RBD_6WAQ_VHH72": "VHH72", "TNFA_5M2J": "anti-TNF"}
        pdb_map = {"RBD_6ZXN_TY1": "6ZXN", "PDL1_5JDS": "5JDS",
                   "TNFA_5M2M": "5M2M", "RBD_6WAQ_VHH72": "6WAQ", "TNFA_5M2J": "5M2J"}
        em = "**" if abs(delta) > 0.1 else ""
        if bl:
            rm = bl["random_iptm_median"]
            rs = bl["random_iptm_std"]
            rand_mean = bl["random_iptm_mean"]
            z = (des - rand_mean) / max(rs, 1e-8)
            md.append(f"| {label} ({pdb_map[tag]}) | {fw_map[tag]} | {em}{wt:.3f}{em} | {em}{des:.3f}{em} | {em}{delta:+.3f}{em} | {rm:.3f}±{rs:.3f} | {z:.1f} |")
    (OUT / "table1_summary.md").write_text("\n".join(md))
    print("Table1 (markdown) saved.")


# ============================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_main_results()
    fig2_trajectories()
    fig3_random_dist()
    fig4_per_target()
    table1_latex()
    print(f"\nAll figures saved to {OUT}/")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")
