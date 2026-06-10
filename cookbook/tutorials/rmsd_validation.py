"""Direct RMSD cross-validation for best designs from v9 pipeline.
Usage: python rmsd_validation.py
Output: /tmp/v9_designs/figures/table2_rmsd.md
"""
from __future__ import annotations
import os, sys, json, time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # must be before torch import
from pathlib import Path
import numpy as np
import torch
import biotite.structure as bs
import biotite.structure.io.pdb as bs_pdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def load_model():
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(MODEL_PATH); config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(MODEL_PATH, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps; _patch_for_mps(model)
    return model

def fold_one(model, binder_seq, target_seq):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
    with torch.inference_mode():
        out = model.forward(**features, num_loops=3, num_sampling_steps=14,
                            num_diffusion_samples=1, calculate_confidence=True)
    return out, features

def kabsch_rmsd(P, Q):
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    cP, cQ = P - P.mean(0), Q - Q.mean(0)
    H = cP.T @ cQ; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T; t = Q.mean(0) - P.mean(0) @ R.T
    return float(np.sqrt(((P @ R.T + t - Q)**2).sum(-1).mean())), R, t

def get_ca_coords(sample_coords, atom_to_token, ref_atom_name_chars, atom_mask, binder_len):
    # All inputs already unbatch (no leading batch dim)
    coords = sample_coords.cpu().numpy()
    if coords.ndim == 3: coords = coords[0]  # (B, N, 3) → (N, 3)
    a2t = atom_to_token.cpu().numpy()
    chars = ref_atom_name_chars.cpu().numpy()
    mask = atom_mask.cpu().numpy().astype(bool)
    binder_ca, target_ca = [], []
    for i in range(len(coords)):
        if not mask[i]: continue
        res = int(a2t[i])
        name = "".join(chr(c+32) if c>0 else " " for c in chars[i]).strip()
        if name != "CA": continue
        if res < binder_len: binder_ca.append((res, coords[i]))
        else: target_ca.append((res, coords[i]))
    binder_ca.sort(key=lambda x: x[0]); target_ca.sort(key=lambda x: x[0])
    return np.array([c for _,c in binder_ca]), np.array([c for _,c in target_ca])

def load_pdb_ca(pdb_path, chain):
    a = bs_pdb.PDBFile.read(pdb_path).get_structure(model=1)
    ca = a[(a.chain_id == chain) & (a.atom_name == "CA")]
    if len(ca) == 0: return np.array([]), ""
    return ca.coord.copy(), "".join(THREE_TO_ONE[r] for r in ca.res_name)

def inject(cdr_subseq, template, cdr_indices):
    chars = list(template)
    for i, c in zip(cdr_indices, cdr_subseq): chars[i] = c
    return "".join(chars)

def resolve(p):
    if os.path.exists(p): return p
    alt = os.path.join(HERE, "..", "..", p)
    return alt if os.path.exists(alt) else p

def main():
    from test_target_pdb import setup_target_design
    summary_data = json.load(open("/tmp/v9_designs/SUMMARY_FULL.json"))
    model = load_model()

    # (label, tag, pdb_design, pdb_rmsd, d_tchain, r_tchain, vhh_chain, fw, epi_str)
    targets = [
        ("Ty1/RBD","RBD_6ZXN_TY1","/tmp/6ZXN_RBD.pdb","/tmp/6ZXN_RBD_Ty1.pdb",
         "A","A","D","ty1","18,19,22,116,117,118,119,120,122,138,140,152,153,154,155,156,159,160,162,163,164"),
        ("KN035/PD-L1","PDL1_5JDS","test/5JDS.pdb","test/5JDS.pdb",
         "A","A","B","kn035","36,38,43,45,48,50,97,98,99,101,102,103,104,105"),
        ("VHH3/TNFα","TNFA_5M2M","test/5M2M.pdb","test/5M2M.pdb",
         "A","B","D","vhh3","15,16,17,18,58,59,60,61,62,63,99,100,130,131,132,133,135"),
        ("VHH72/RBD","RBD_6WAQ_VHH72","test/6WAQ.pdb","test/6WAQ.pdb",
         "B","B","A","vhh72","35,36,37,38,39,40,41,42,43,44,45,46,49,50"),
        ("anti-TNF","TNFA_5M2J","test/5M2J.pdb","test/5M2J.pdb",
         "A","A","D","antitnf","66,67,68,79,80,81,82,83,84,117,118"),
    ]

    results = []
    for label, tag, pdb_d, pdb_r, d_tchain, r_tchain, vhh_chain, fw, epi_str in targets:
        epi = [int(x) for x in epi_str.split(",") if x.strip()]
        dp, rp = resolve(pdb_d), resolve(pdb_r)
        setup = setup_target_design(pdb_path=dp, target_chain=d_tchain, epitope_indices=epi, framework=fw)
        target_seq, binder_template, binder_wt = setup["target_sequence"], setup["binder_template"], setup["binder_full_sequence"]
        cdr_idx = setup["cdr_indices"]; binder_len = len(binder_wt)

        ev = summary_data.get(tag, {})
        best_iptm, best_cdr = -1.0, None
        for sr in ev.get("per_seed", []):
            for e in sr.get("evals", []):
                if e.get("iptm") is not None and e["iptm"] > best_iptm:
                    best_iptm = e["iptm"]; best_cdr = e["cdr_seq"]
        if best_cdr is None: print(f"[SKIP] {label}"); continue
        best_full = inject(best_cdr, binder_template, cdr_idx)
        real_target_ca, _ = load_pdb_ca(rp, r_tchain)
        real_vhh_ca, _ = load_pdb_ca(rp, vhh_chain)
        print(f"\n=== {label} === real target={len(real_target_ca)}CA VHH={len(real_vhh_ca)}CA")
        print(f"  WT: {''.join(binder_wt[i] for i in cdr_idx)[:40]}")
        print(f"  DES: {best_cdr[:40]}  (ipTM={best_iptm:.3f})")

        row = {"label":label, "wt_iptm":ev.get("wt_iptm",0), "design_iptm":best_iptm}
        for name, seq in [("WT", binder_wt), ("Design", best_full)]:
            t0=time.time(); out, feats = fold_one(model, seq, target_seq); dt=time.time()-t0
            sc=out["sample_atom_coords"].float()
            if sc.dim()==4: sc=sc[:,0]
            # Handle optional batch dim
            a2t = feats["atom_to_token"]
            anc = feats["ref_atom_name_chars"]
            am  = feats["atom_attention_mask"]
            a2t_0 = a2t[0] if a2t.dim()>1 else a2t
            anc_0 = anc[0] if anc.dim()>1 else anc
            am_0  = am[0] if am.dim()>1 else am
            b_ca, t_ca = get_ca_coords(sc, a2t_0, anc_0, am_0, binder_len)
            n_t=min(len(t_ca), len(real_target_ca)); n_b=min(len(b_ca), len(real_vhh_ca))
            if n_t<3 or n_b<3:
                print(f"  {name}: SKIP (too few coords: t={n_t}, b={n_b})")
                continue
            t_rmsd, R, tvec = kabsch_rmsd(t_ca[:n_t], real_target_ca[:n_t])
            b_aligned = b_ca[:n_b] @ R.T + tvec
            b_rmsd = float(np.sqrt(((b_aligned - real_vhh_ca[:n_b])**2).sum(-1).mean()))
            pd=np.linalg.norm(b_aligned[:,None,:]-real_target_ca[:n_t][None,:,:],axis=-1)
            n_iface=int((pd<8.0).sum())
            rd=np.linalg.norm(real_vhh_ca[:n_b][:,None,:]-real_target_ca[:n_t][None,:,:],axis=-1)
            n_real_iface=int((rd<8.0).sum())
            iptm=float(out["iptm"][0]); ptm=float(out["ptm"][0])
            print(f"  {name:>6s}: ipTM={iptm:.3f} pTM={ptm:.3f} tRMSD={t_rmsd:.2f}Å bRMSD={b_rmsd:.2f}Å cont={n_iface}/{n_real_iface} [{dt:.0f}s]")
            row[f"{name.lower()}_iptm"]=iptm; row[f"{name.lower()}_ptm"]=ptm
            row[f"{name.lower()}_brmsd"]=b_rmsd; row[f"{name.lower()}_trmsd"]=t_rmsd
            row[f"{name.lower()}_contacts"]=n_iface
        results.append(row)

    OUT = Path("/tmp/v9_designs/figures")
    lines=["| Target | WT ipTM | Design ipTM | WT bRMSD | Design bRMSD | Δ bRMSD | WT cont | Des cont | Real cont |"]
    lines.append("|--------|---------|-------------|----------|--------------|---------|---------|----------|-----------|")
    for r in results:
        dbr = r.get('design_brmsd', 0) - r.get('wt_brmsd', 0)
        lines.append(f"| {r['label']} | {r.get('wt_iptm',0):.3f} | {r.get('design_iptm',0):.3f} | "
                     f"{r.get('wt_brmsd',0):.2f} | {r.get('design_brmsd',0):.2f} | {dbr:+.2f} | "
                     f"{r.get('wt_contacts',0)} | {r.get('design_contacts',0)} | {r.get('wt_contacts',0)} |")
    (OUT/"table2_rmsd.md").write_text("\n".join(lines))
    json.dump(results, open(OUT/"rmsd_results.json","w"), indent=2)
    print(f"\nSaved to {OUT/'table2_rmsd.md'}")

if __name__ == "__main__": main()
