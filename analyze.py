"""Supplementary analyses for EasyNano designs.

1. Fast vs Full model discrepancy (Supplementary Table S2)
2. Developability assessment (aggregation, PTM, pI)
3. Multi-seed robustness check
"""

from __future__ import annotations

import json
import numpy as np
from collections import Counter

# ---- AA properties ----
# Kyte-Doolittle hydrophobicity
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# pKa values for charged residues
PKA = {"D": 3.9, "E": 4.3, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
N_TERM_PKA = 8.0
C_TERM_PKA = 3.5

# PTM risk motifs
DEAMIDATION_MOTIFS = ["NG", "NS", "NN", "QG", "QS"]  # NG is the strongest signal
OXIDATION_MOTIFS = ["M"]  # Met oxidation
ISOMERIZATION_MOTIFS = ["DG", "DS", "DP"]  # Asp isomerization
PROTEOLYSIS_MOTIFS = ["DP", "KP", "RP"]  # Proline-directed cleavage


def compute_gravy(sequence: str) -> float:
    """Grand average of hydropathy (GRAVY). Positive = hydrophobic."""
    vals = [HYDROPATHY.get(aa, 0.0) for aa in sequence]
    return sum(vals) / len(vals) if vals else 0.0


def compute_net_charge(sequence: str, ph: float = 7.4) -> float:
    """Estimate net charge at given pH using simple pKa model."""
    charge = 0.0
    # N-terminus
    charge += 1.0 / (1.0 + 10 ** (ph - N_TERM_PKA))
    # C-terminus
    charge += -1.0 / (1.0 + 10 ** (C_TERM_PKA - ph))
    for aa in sequence:
        if aa in PKA:
            if aa in ("D", "E", "C", "Y"):
                charge += -1.0 / (1.0 + 10 ** (PKA[aa] - ph))
            elif aa in ("R", "K"):
                charge += 1.0 / (1.0 + 10 ** (ph - PKA[aa]))
            elif aa == "H":
                charge += 1.0 / (1.0 + 10 ** (ph - PKA[aa]))
    return charge


def find_ptm_motifs(sequence: str) -> dict:
    """Scan for common PTM risk motifs."""
    motifs = {
        "deamidation": [],
        "oxidation": [],
        "isomerization": [],
        "proteolysis": [],
    }
    for i in range(len(sequence) - 1):
        dipep = sequence[i:i + 2]
        if dipep in DEAMIDATION_MOTIFS:
            motifs["deamidation"].append((i, dipep))
        if dipep in ISOMERIZATION_MOTIFS:
            motifs["isomerization"].append((i, dipep))
        if dipep in PROTEOLYSIS_MOTIFS:
            motifs["proteolysis"].append((i, dipep))
    for i, aa in enumerate(sequence):
        if aa in OXIDATION_MOTIFS:
            motifs["oxidation"].append((i, aa))
    return motifs


def developability_report(sequence: str, cdr_indices: list[int] | None = None,
                          label: str = "") -> dict:
    """Compute developability metrics for a binder sequence.

    Parameters
    ----------
    sequence : str
        Full binder amino acid sequence.
    cdr_indices : list[int] or None
        If provided, metrics are also computed on CDR-only subset.

    Returns dict of metrics.
    """
    gravy = compute_gravy(sequence)
    charge = compute_net_charge(sequence)
    ptm = find_ptm_motifs(sequence)
    aa_counts = Counter(sequence)

    report = {
        "label": label,
        "length": len(sequence),
        "gravy": round(gravy, 3),
        "net_charge_pH7.4": round(charge, 2),
        "n_cys": aa_counts.get("C", 0),
        "n_met": aa_counts.get("M", 0),
        "n_trp": aa_counts.get("W", 0),
        "aromatic_pct": round(
            (aa_counts.get("F", 0) + aa_counts.get("Y", 0) +
             aa_counts.get("W", 0) + aa_counts.get("H", 0)) / len(sequence) * 100, 1),
        "n_deamidation_sites": len(ptm["deamidation"]),
        "n_oxidation_sites": len(ptm["oxidation"]),
        "n_isomerization_sites": len(ptm["isomerization"]),
        "n_proteolysis_sites": len(ptm["proteolysis"]),
    }

    if ptm["deamidation"]:
        report["deamidation_sites"] = [
            f"{motif}@{pos}" for pos, motif in ptm["deamidation"]]
    if ptm["isomerization"]:
        report["isomerization_sites"] = [
            f"{motif}@{pos}" for pos, motif in ptm["isomerization"]]

    if cdr_indices:
        cdr_seq = "".join(sequence[i] for i in cdr_indices)
        report["cdr_only"] = {
            "sequence": cdr_seq,
            "gravy": round(compute_gravy(cdr_seq), 3),
            "net_charge": round(compute_net_charge(cdr_seq), 2),
            "n_met": cdr_seq.count("M"),
            "n_trp": cdr_seq.count("W"),
        }

    return report


def print_developability_report(report: dict):
    """Pretty-print a developability report."""
    print(f"\n  {'─'*60}")
    print(f"  Developability: {report['label']}")
    print(f"  {'─'*60}")
    print(f"  Length:              {report['length']}")
    print(f"  GRAVY:               {report['gravy']}  "
          f"({'hydrophobic' if report['gravy'] > 0 else 'hydrophilic'})")
    print(f"  Net charge (pH 7.4): {report['net_charge_pH7.4']:+.1f}")
    print(f"  Aromatic %:          {report['aromatic_pct']}%")
    print(f"  Cysteine:            {report['n_cys']}")
    print(f"  Methionine:          {report['n_met']}")
    print(f"  Tryptophan:          {report['n_trp']}")
    print(f"  Deamidation sites:   {report['n_deamidation_sites']}"
          f"{' ' + str(report.get('deamidation_sites', '')) if report['n_deamidation_sites'] else ''}")
    print(f"  Oxidation sites:     {report['n_oxidation_sites']}")
    print(f"  Isomerization sites: {report['n_isomerization_sites']}"
          f"{' ' + str(report.get('isomerization_sites', '')) if report['n_isomerization_sites'] else ''}")
    print(f"  Proteolysis sites:   {report['n_proteolysis_sites']}")

    flags = []
    if report['n_cys'] > 0 and report['n_cys'] % 2 != 0:
        flags.append("UNPAIRED CYS")
    if report['n_deamidation_sites'] > 2:
        flags.append("DEAMIDATION RISK")
    if report['n_oxidation_sites'] > 3:
        flags.append("OXIDATION RISK")
    if report['n_isomerization_sites'] > 1:
        flags.append("ISOMERIZATION RISK")
    if report['gravy'] > 0.5:
        flags.append("HYDROPHOBIC (aggregation risk)")
    if report['gravy'] < -1.0:
        flags.append("VERY HYDROPHILIC (stability risk)")
    if flags:
        print(f"  \033[93mFLAGS: {', '.join(flags)}\033[0m")
    else:
        print(f"  No developability flags.")

    if "cdr_only" in report:
        cdr = report["cdr_only"]
        print(f"\n  CDR-only:")
        print(f"    GRAVY: {cdr['gravy']}  "
              f"Charge: {cdr['net_charge']:+.1f}  "
              f"Met: {cdr['n_met']}  Trp: {cdr['n_trp']}")
        print(f"    Sequence: {cdr['sequence']}")


def fast_vs_full_table(targets_data: list[dict]) -> str:
    """Build Supplementary Table S2: Fast vs Full model discrepancy.

    Each entry in targets_data should have:
        target, fast_iptm, full_iptm, fast_cdr_epi, full_cdr_epi
    """
    rows = []
    for d in targets_data:
        diptm = d["full_iptm"] - d["fast_iptm"]
        depi = d["full_cdr_epi"] - d["fast_cdr_epi"]
        rows.append(
            f"  {d['target']:<20} {d['fast_iptm']:>8.3f} {d['full_iptm']:>8.3f} "
            f"{diptm:>+8.3f} {d['fast_cdr_epi']:>10.1f} {d['full_cdr_epi']:>10.1f} "
            f"{depi:>+8.1f}"
        )

    header = (f"  {'Target':<20} {'Fast ipTM':>8} {'Full ipTM':>8} "
              f"{'ΔipTM':>8} {'Fast cd→epi':>10} {'Full cd→epi':>10} {'Δcd→epi':>8}")
    return "\n".join([header, f"  {'-'*78}"] + rows)
