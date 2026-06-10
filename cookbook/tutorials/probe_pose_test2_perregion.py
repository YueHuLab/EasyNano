"""Re-analyze Test 2 results with proper per-region alignment.

The previous test 2 had a bug: per-region RMSDs (framework=11Å, CDR=5.6Å)
were inconsistent with the overall 1.38Å binder RMSD. The bug was that
the Kabsch was computed on target, but the per-region was compared on
unaligned binder.

This script re-aligns on the BINDER itself for per-region comparison.
"""
from __future__ import annotations

import numpy as np
import json
from pathlib import Path

BINDER_LEN = 127
out_dir = Path("/tmp/b5_pose_test2")
init_b = np.load(out_dir / "init_binder_ca.npy")
init_t = np.load(out_dir / "init_target_ca.npy")
best_b = np.load(out_dir / "best_binder_ca.npy")
best_t = np.load(out_dir / "best_target_ca.npy")

CDR = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       54, 55, 56, 57, 58, 59,
       101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
fw = [i for i in range(BINDER_LEN) if i not in set(CDR)]


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_aligned = (Pc @ R.T) + Q.mean(0, keepdims=True)
    diff = P_aligned - Q
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def kabsch_rotation(P, Q):
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


print("=" * 60)
print("Test 2: v9_init vs v9_best_15seed (same framework, 5 CDR diffs)")
print("=" * 60)

# A. Target-alignment: most natural pose comparison
offset = best_t.mean(0) - init_t.mean(0)
init_b_ta = init_b + offset
total_ta = kabsch_rmsd(init_b_ta, best_b)
fw_ta = kabsch_rmsd(init_b_ta[fw], best_b[fw])
cdr_ta = kabsch_rmsd(init_b_ta[CDR], best_b[CDR])
h1_ta = kabsch_rmsd(init_b_ta[CDR[:10]], best_b[CDR[:10]])
h2_ta = kabsch_rmsd(init_b_ta[CDR[10:16]], best_b[CDR[10:16]])
h3_ta = kabsch_rmsd(init_b_ta[CDR[16:]], best_b[CDR[16:]])

print(f"\n  A. Target-mean alignment (offset only, no rotation):")
print(f"    Total binder (127):  {total_ta:.2f}Å")
print(f"    Framework (95):      {fw_ta:.2f}Å")
print(f"    CDR total (32):      {cdr_ta:.2f}Å")
print(f"    H1 (10):             {h1_ta:.2f}Å")
print(f"    H2 (6):              {h2_ta:.2f}Å")
print(f"    H3 (16):             {h3_ta:.2f}Å")

# B. Kabsch alignment on FULL binder — measures shape difference, removes rigid-body
R = kabsch_rotation(init_b, best_b)
init_b_kab = (init_b - init_b.mean(0)) @ R.T + best_b.mean(0)
total_b = kabsch_rmsd(init_b_kab, best_b)
fw_b = kabsch_rmsd(init_b_kab[fw], best_b[fw])
cdr_b = kabsch_rmsd(init_b_kab[CDR], best_b[CDR])
h1_b = kabsch_rmsd(init_b_kab[CDR[:10]], best_b[CDR[:10]])
h2_b = kabsch_rmsd(init_b_kab[CDR[10:16]], best_b[CDR[10:16]])
h3_b = kabsch_rmsd(init_b_kab[CDR[16:]], best_b[CDR[16:]])

print(f"\n  B. Kabsch alignment on FULL binder (rigid-body removed):")
print(f"    Total binder (127):  {total_b:.2f}Å")
print(f"    Framework (95):      {fw_b:.2f}Å")
print(f"    CDR total (32):      {cdr_b:.2f}Å")
print(f"    H1 (10):             {h1_b:.2f}Å")
print(f"    H2 (6):              {h2_b:.2f}Å")
print(f"    H3 (16):             {h3_b:.2f}Å")

# C. Kabsch on framework only — measure CDR-only deviation
R_fw = kabsch_rotation(init_b[fw], best_b[fw])
init_b_fw = (init_b - init_b[fw].mean(0)) @ R_fw.T + best_b[fw].mean(0)
total_fw = kabsch_rmsd(init_b_fw, best_b)
fw_fw = kabsch_rmsd(init_b_fw[fw], best_b[fw])
cdr_fw = kabsch_rmsd(init_b_fw[CDR], best_b[CDR])

print(f"\n  C. Kabsch alignment on FRAMEWORK only (CDR positions are the 'free' part):")
print(f"    Total binder (127):  {total_fw:.2f}Å")
print(f"    Framework (95):      {fw_fw:.2f}Å   (should be ~0 by construction)")
print(f"    CDR total (32):      {cdr_fw:.2f}Å   (this is the actual CDR deviation)")

# Interface centroid
EPI = [24, 30, 31, 33, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
       173, 174, 176, 177, 196, 197]
e_init = init_b[CDR].mean(0) - init_t[EPI].mean(0)
e_best = best_b[CDR].mean(0) - best_t[EPI].mean(0)
print(f"\n  Interface centroid (CDR-mean - epitope-mean):")
print(f"    v9_init: |{e_init}| = {np.linalg.norm(e_init):.2f}Å")
print(f"    v9_best: |{e_best}| = {np.linalg.norm(e_best):.2f}Å")
print(f"    |e_init - e_best| = {np.linalg.norm(e_init - e_best):.2f}Å")
