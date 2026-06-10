"""Generate CDR sequence logos for WT and best designs.
Output: /tmp/v9_designs/figures/cdr_logos/
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

OUT = Path("/tmp/v9_designs/figures/cdr_logos")
OUT.mkdir(parents=True, exist_ok=True)

AA20 = "ARNDCQEGHILKMFPSTWYV"

summary = json.load(open("/tmp/v9_designs/SUMMARY_FULL.json"))

TARGETS = [
    ("RBD_6ZXN_TY1", "Ty1/RBD", "#2196F3"),
    ("PDL1_5JDS", "KN035/PD-L1", "#4CAF50"),
]

def get_wt_cdr(tag):
    """Extract WT CDR from first snapshot's init_cdr."""
    for seed in [0,1,2]:
        p = f"/tmp/v9_designs/{tag}_seed{seed}_snaps.json"
        if os.path.exists(p):
            d = json.load(open(p))
            return d.get("init_cdr", "")
    return ""

def get_best_cdr(tag):
    ev = summary.get(tag, {})
    best_iptm, best_cdr = -1.0, None
    for sr in ev.get("per_seed", []):
        for e in sr.get("evals", []):
            if e.get("iptm") is not None and e["iptm"] > best_iptm:
                best_iptm = e["iptm"]; best_cdr = e["cdr_seq"]
    return best_cdr or ""

def seq_to_matrix(seq):
    """Convert AA sequence to DataFrame for logomaker."""
    mat = np.zeros((len(seq), len(AA20)))
    for i, aa in enumerate(seq):
        if aa in AA20:
            mat[i, AA20.index(aa)] = 1.0
    return pd.DataFrame(mat, columns=list(AA20))

for tag, label, color in TARGETS:
    wt_cdr = get_wt_cdr(tag)
    des_cdr = get_best_cdr(tag)
    if not wt_cdr or not des_cdr:
        continue

    # Trim to same length
    n = min(len(wt_cdr), len(des_cdr))
    wt_cdr = wt_cdr[:n]
    des_cdr = des_cdr[:n]

    wt_mat = seq_to_matrix(wt_cdr)
    des_mat = seq_to_matrix(des_cdr)

    # Find differing positions
    diff_pos = [i for i in range(n) if wt_cdr[i] != des_cdr[i]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(n*0.4, 8), 4.5))

    # WT logo
    logo_wt = logomaker.Logo(wt_mat, ax=ax1, color_scheme="chemistry")
    logo_wt.style_xticks(anchor=0, spacing=1)
    ax1.set_ylabel("WT", fontweight="bold", fontsize=11)
    ax1.set_title(f"{label} — CDR Sequence", fontweight="bold", fontsize=12)

    # Design logo with mutations highlighted
    logo_des = logomaker.Logo(des_mat, ax=ax2, color_scheme="chemistry")
    logo_des.style_xticks(anchor=0, spacing=1)
    # Highlight mutated positions
    for i in diff_pos:
        ax2.axvspan(i-0.4, i+0.4, facecolor="red", alpha=0.15, edgecolor="red", linewidth=0.5)
    ax2.set_ylabel("Design", fontweight="bold", fontsize=11)
    ax2.set_xlabel(f"CDR Position  ({len(diff_pos)}/{n} mutated, highlighted in red)")

    # Add ipTM annotation
    ev = summary.get(tag, {})
    wt_iptm = ev.get("wt_iptm", 0)
    des_iptm = ev.get("best_design_iptm") or 0
    fig.text(0.99, 0.98, f"WT ipTM={wt_iptm:.3f}  →  Design ipTM={des_iptm:.3f}  (Δ={des_iptm-wt_iptm:+.3f})",
             transform=fig.transFigure, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    safe = tag.replace("/","_")
    fig.savefig(OUT / f"{safe}_cdr_logo.pdf")
    fig.savefig(OUT / f"{safe}_cdr_logo.png", dpi=200)
    plt.close()
    print(f"  {label}: {len(diff_pos)}/{n} mutations highlighted")

# Also make a combined figure for all targets
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
all_tags = [
    ("RBD_6ZXN_TY1", "Ty1/RBD", "#2196F3"),
    ("PDL1_5JDS", "KN035/PD-L1", "#4CAF50"),
    ("TNFA_5M2M", "VHH3/TNFα", "#FF9800"),
    ("RBD_6WAQ_VHH72", "VHH72/RBD", "#9C27B0"),
    ("TNFA_5M2J", "anti-TNF", "#F44336"),
]
for idx, (tag, label, color) in enumerate(all_tags):
    ax = axes[idx]
    wt = get_wt_cdr(tag)
    des = get_best_cdr(tag)
    if not wt or not des:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); continue
    n = min(len(wt), len(des))
    wt, des = wt[:n], des[:n]
    diff = sum(1 for i in range(n) if wt[i] != des[i])
    des_mat = seq_to_matrix(des)
    logo = logomaker.Logo(des_mat, ax=ax, color_scheme="chemistry")
    logo.style_xticks(anchor=0, spacing=1)
    for i in range(n):
        if wt[i] != des[i]:
            ax.axvspan(i-0.4, i+0.4, facecolor="red", alpha=0.12, edgecolor="red", linewidth=0.3)
    ev = summary.get(tag, {})
    wt_i = ev.get("wt_iptm", 0); des_i = ev.get("best_design_iptm") or 0
    ax.set_ylabel(label, fontweight="bold", fontsize=9)
    ax.text(0.99, 0.85, f"{wt_i:.3f}→{des_i:.3f} (Δ{des_i-wt_i:+.3f})  {diff}/{n} mut",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

axes[0].set_title("CDR Sequence Logos — WT vs Best Design (mutations highlighted in red)", fontsize=13, fontweight="bold")
axes[-1].set_xlabel("CDR Position")
plt.tight_layout()
fig.savefig(OUT / "all_cdr_logos.pdf")
fig.savefig(OUT / "all_cdr_logos.png", dpi=200)
plt.close()

print(f"\nAll logos saved to {OUT}/")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
