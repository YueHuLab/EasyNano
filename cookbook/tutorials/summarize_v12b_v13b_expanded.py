"""Final aggregation: combine v12b/v13b + expanded multi-start results,
print top-10 by iptm, top-10 by CDR→epi, identify the new global best.
"""
import json
from pathlib import Path

NEW_BEST_CDR = "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR"
V9_STEP48_CDR = "GLQIGYGWYMSYSGQRRVVADSPQRIYKAPIR"

def load_eval(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)

def main():
    v12b = load_eval("/tmp/b5_v12b_eval.json")
    v13b = load_eval("/tmp/b5_v13b_eval.json")
    ms_orig = load_eval("/tmp/b5_v9_multistart_eval.json")
    ms_exp = load_eval("/tmp/b5_v9_multistart_expanded_eval.json")

    all_results = v12b + v13b + ms_orig + ms_exp

    # Filter to entries with non-null iptm
    valid = [r for r in all_results if r.get("iptm") is not None]
    print(f"Total valid results: {len(valid)}")
    print(f"  v12b (Full in loop from new best): {len(v12b)}")
    print(f"  v13b (H3-only from new best):      {len(v13b)}")
    print(f"  multi-start seeds 1-5:             {len(ms_orig)}")
    print(f"  multi-start seeds 6-15:            {len(ms_exp)}")

    # Baseline references
    print(f"\nBaselines:")
    print(f"  v9 step 48 (old best):   iptm=0.661 pTM=0.843 CDR→epi=10.02 Å  {V9_STEP48_CDR}")
    print(f"  seed=2 step 56 (new best): iptm=0.717 pTM=0.853 CDR→epi=9.52 Å  {NEW_BEST_CDR}")

    # Top 15 by iptm
    print(f"\n=== Top 15 by ipTM ===")
    top_iptm = sorted(valid, key=lambda r: -r["iptm"])[:15]
    for r in top_iptm:
        name = r["name"]
        iptm = r["iptm"]
        ptm = r.get("ptm") or 0
        cdr = r.get("cdr_to_epi_min", 0)
        cdr_seq = r.get("cdr_seq", "")
        print(f"  {name:>40}  iptm={iptm:.3f}  pTM={ptm:.3f}  CDR→epi={cdr:.2f}  {cdr_seq}")

    # Top 15 by CDR→epi
    print(f"\n=== Top 15 by CDR→epi (closest to epitope) ===")
    top_epi = sorted(valid, key=lambda r: r["cdr_to_epi_min"])[:15]
    for r in top_epi:
        name = r["name"]
        iptm = r.get("iptm") or 0
        ptm = r.get("ptm") or 0
        cdr = r.get("cdr_to_epi_min", 0)
        cdr_seq = r.get("cdr_seq", "")
        print(f"  {name:>40}  CDR→epi={cdr:.2f}Å  iptm={iptm:.3f}  pTM={ptm:.3f}  {cdr_seq}")

    # Identify new global best
    best = max(valid, key=lambda r: r["iptm"])
    print(f"\n=== Global best (across all directions) ===")
    print(f"  name:   {best['name']}")
    print(f"  iptm:   {best['iptm']:.4f}")
    print(f"  pTM:    {best['ptm']:.4f}")
    print(f"  CDR→epi: {best['cdr_to_epi_min']:.2f} Å")
    print(f"  CDR:    {best.get('cdr_seq', '')}")
    print(f"  full binder: {best.get('full_seq', '')}")

if __name__ == "__main__":
    main()
