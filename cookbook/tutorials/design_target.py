"""Generalized v2 epitope-targeted binder design loop.

Drop-in replacement for design_b5_mps_v2.py that takes an arbitrary
target PDB + epitope, instead of being hardcoded to B5.

Pipeline:
  1. test_target_pdb.setup_target_design(target_pdb, target_chain, epitope, framework)
     -> target sequence, binder template, CDRs, epitope, prior
  2. Init soft logits (CDR positions near WT AA with noise)
  3. Adam loop on (intra, inter, glob, epitope, structure_prior) losses,
     with explicit pinning of framework positions.
  4. Save CDR snapshots every SNAPSHOT_EVERY steps.
  5. Print summary: best-by-inter, best-by-epi, best-by-total.

By default uses ESMFold2-Fast (721M) for the design loop (gradients
flow well) and leaves Full ESMFold2 (1.3G) for the separate eval
script (eval_snapshots_v2.py).

Usage:
    python design_target.py \\
        --target-pdb test/5JDS.pdb --target-chain A \\
        --epitope-indices 36,38,43,45,48,50,97,98,99,101,102,103,104,105 \\
        --framework b5 --seed 0 --steps 100
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from binder_design_hy_losses import (  # noqa: E402
    LOSS_WEIGHTS, MUTABLE_TOKEN,
    compute_structure_losses, get_mid_points,
)
from test_target_pdb import setup_target_design  # noqa: E402

# Use Fast ESMFold2 (721M) for design — gradients are larger than Full.
MODEL_PATH = "/Users/huyue/esm-c-fold2/ESMFold2-Fast"
ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
DEVICE = "mps"
NUM_RES_TYPES = 33

TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
AA_DIMS = 20
CYS_TOK = AA_TO_TOKEN["C"]  # 6

# Natural amino acid frequency (UniProt background) — used as cheap LM prior
AA_FREQ = torch.tensor([
    0.0743, 0.0510, 0.0443, 0.0477, 0.0290, 0.0399, 0.0604, 0.0677,
    0.0227, 0.0554, 0.0968, 0.0580, 0.0221, 0.0394, 0.0444, 0.0580,
    0.0537, 0.0127, 0.0300, 0.0660,
])

# Design defaults (same as design_b5_mps_v2.py for cross-version comparability)
N_STEPS = 100
LR = 0.5
TEMP_MIN = 0.1
LOG_EVERY = 5
SNAPSHOT_EVERY = 10
SAMPLE_STEPS_FWD = 5
N_LOOPS_FWD = 1
W_AA_FREQ = 0.01


def load_model(model_path: str = MODEL_PATH):
    print(f"Loading ESMFold2 from {model_path} ...")
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained(model_path)
    config.esmc_id = ESMC_PATH
    model = ESMFold2Model.from_pretrained(model_path, config=config).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    for p in model.parameters():
        p.requires_grad_(False)
    unwrapped = model.forward
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
    model.forward = unwrapped.__get__(model, type(model))
    print(f"  loaded in {time.time() - t0:.1f}s, params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


def init_soft_logits(binder_template: str, binder_wt: str, wt_logit: float = 3.0,
                     pin_cys_in_cdr: bool = True) -> torch.Tensor:
    """Initialize soft logits for binder design.

    pin_cys_in_cdr: when True, pin Cys to -10 at all CDR positions (B5
        convention; safe for B5 H3 which has no Cys, avoids spurious
        disulfides). When False, allow Cys in CDRs (needed for VHH
        frameworks like KN035 whose H3 contains a structural disulfide).
    """
    L = len(binder_template)
    logits = torch.zeros(L, AA_DIMS)
    for i, aa in enumerate(binder_template):
        if aa != MUTABLE_TOKEN:
            assert aa in TOKENS[2:22]
            idx = AA_TO_TOKEN[aa] - 2
            logits[i, :] = -10.0
            logits[i, idx] = 10.0
        else:
            wt_aa = binder_wt[i]
            assert wt_aa in TOKENS[2:22]
            wt_idx = AA_TO_TOKEN[wt_aa] - 2
            logits[i, :] = 0.5 * torch.randn(AA_DIMS)
            logits[i, wt_idx] = wt_logit
            if pin_cys_in_cdr:
                logits[i, CYS_TOK - 2] = -10.0
    return logits.requires_grad_(True)


def pin_fixed_positions(soft_logits: torch.Tensor, binder_template: str):
    with torch.no_grad():
        for i, aa in enumerate(binder_template):
            if aa != MUTABLE_TOKEN:
                idx = AA_TO_TOKEN[aa] - 2
                soft_logits[i, :] = -10.0
                soft_logits[i, idx] = 10.0


def fixed_position_mask(binder_template: str, device) -> torch.Tensor:
    return torch.tensor(
        [aa != MUTABLE_TOKEN for aa in binder_template],
        dtype=torch.bool, device=device
    )


def build_soft_res_type(soft_logits: torch.Tensor, target_one_hot: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
    binder_probs_20 = F.softmax(soft_logits / max(temperature, 1e-3), dim=-1)
    binder_probs_33 = torch.zeros(
        soft_logits.size(0), NUM_RES_TYPES,
        device=soft_logits.device, dtype=binder_probs_20.dtype
    )
    binder_probs_33[:, 2:22] = binder_probs_20
    binder_probs_33 = binder_probs_33.unsqueeze(0)
    return torch.cat([binder_probs_33, target_one_hot.to(binder_probs_33.device)], dim=1)


def make_target_one_hot(target_seq: str, device) -> torch.Tensor:
    L = len(target_seq)
    idx = torch.tensor([AA_TO_TOKEN[aa] for aa in target_seq], device=device).long()
    oh = F.one_hot(idx, num_classes=NUM_RES_TYPES).float()
    return oh.unsqueeze(0)


def soft_to_hard_seq(soft_logits: torch.Tensor) -> str:
    idx = soft_logits.argmax(-1).cpu().tolist()
    return "".join(TOKENS[i + 2] for i in idx)


def cdr_to_epitope_stats(disto_logits: torch.Tensor, cdr_indices: list[int],
                         epitope_target_indices: list[int],
                         target_length: int, binder_length: int) -> dict:
    midpoints = get_mid_points().to(disto_logits.device)
    probs = torch.softmax(disto_logits, dim=-1)
    e_dist = (probs * midpoints).sum(-1)[0]
    cross = e_dist[target_length:, :target_length]
    cdr_rows = [b for b in cdr_indices]
    cdr_to_e = cross[cdr_rows][:, epitope_target_indices]
    return {
        "cdr_to_epitope_min": cdr_to_e.min(dim=-1).values.mean().item(),
        "cdr_to_epitope_median": cdr_to_e.min(dim=-1).values.median().item(),
        "inter_min": cross.min().item(),
        "inter_median": cross.median().item(),
    }


def aa_freq_loss(soft_logits: torch.Tensor, mutable_mask: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(soft_logits, dim=-1)
    log_freq = torch.log(AA_FREQ.to(probs.device))
    expected_log = (probs * log_freq.unsqueeze(0)).sum(-1)
    nll = -expected_log * mutable_mask.float()
    return nll.sum() / (mutable_mask.sum() + 1e-8)


def reorder_bf_to_target_first(disto_bf: torch.Tensor, binder_len: int) -> torch.Tensor:
    L = disto_bf.size(1)
    perm = torch.cat([torch.arange(binder_len, L),
                      torch.arange(0, binder_len)])
    return disto_bf[:, perm, :, :][:, :, perm, :]


def align_prior_to_disto(prior_bins, prior_mask, disto_target_first):
    L_p = prior_bins.size(0)
    L_d = disto_target_first.size(1)
    if L_d == L_p:
        return prior_bins, prior_mask
    if L_d < L_p:
        raise RuntimeError(f"Distogram L={L_d} < prior L={L_p}; cannot pad prior.")
    return prior_bins[:L_p, :L_p], prior_mask[:L_p, :L_p]


def run_design(
    pdb_path: str,
    target_chain: str,
    epitope_indices: list[int],
    framework: str = "b5",
    steps: int = N_STEPS,
    lr: float = LR,
    seed: int = 0,
    log_every: int = LOG_EVERY,
    snapshot_every: int = SNAPSHOT_EVERY,
    sample_steps: int = SAMPLE_STEPS_FWD,
    n_loops: int = N_LOOPS_FWD,
    w_aa_freq: float = W_AA_FREQ,
    wt_logit: float = 3.0,
    w_epitope: float = 0.05,
    w_intra: float = 0.5,
    w_inter: float = 0.5,
    w_glob: float = 0.2,
    w_prior: float = 0.3,
    snapshot_path: str = "/tmp/target_design_snapshots.json",
    use_full_model: bool = False,
    pin_cys_in_cdr: bool = True,
    binder_chain: str | None = None,
):
    print(f"=== Target {pdb_path} chain {target_chain} framework {framework} ===")
    print(f"=== Epitope: {epitope_indices} ===\n")

    setup = setup_target_design(
        pdb_path=pdb_path,
        target_chain=target_chain,
        epitope_indices=epitope_indices,
        framework=framework,
        binder_chain=binder_chain,
    )
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    L_prior = prior_bins.size(0)

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}, Prior L={L_prior}")

    model_path = ("/Users/huyue/esm-c-fold2/ESMFold2"
                  if use_full_model else MODEL_PATH)
    model = load_model(model_path=model_path)
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    binder_wt = setup["binder_full_sequence"]
    import binder_design_hy_losses as L
    L.LOSS_WEIGHTS["epitope"] = w_epitope
    L.LOSS_WEIGHTS["intra_contact"] = w_intra
    L.LOSS_WEIGHTS["inter_contact"] = w_inter
    L.LOSS_WEIGHTS["glob"] = w_glob
    L.LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} intra={w_intra} inter={w_inter} "
          f"glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit,
                                    pin_cys_in_cdr=pin_cys_in_cdr).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    history = []
    snapshots = []
    best_inter = float("inf")
    best_seq_inter = ""
    best_step_inter = -1
    best_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1

    init_seq = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_seq[i] for i in cdr)
    print(f"\n  init CDRs: {init_cdr}\n")

    print(f"Designing {steps} steps (lr={lr}, sample={sample_steps}, "
          f"loops={n_loops}, w_aa_freq={w_aa_freq}) ...")
    header = (f"  {'step':>4}  {'total':>8}  {'intra':>7}  {'inter':>7}  "
              f"{'glob':>7}  {'epi':>7}  {'prior':>7}  "
              f"{'CDR→epi':>8}  {'inter_min':>9}  "
              f"{'pTM':>5}  {'ipTM':>5}  CDR_seq")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    t_start = time.time()
    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)

        cur_seq = soft_to_hard_seq(soft_logits)
        from esmscore._complex import build_complex_features
        feats = build_complex_features(cur_seq, target_seq)
        features = {k: v for k, v in feats.items() if not k.startswith("_")}
        features["res_type"] = res_type_soft.to(DEVICE)
        features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}

        with torch.set_grad_enabled(True):
            out = model.forward(
                **features,
                num_loops=n_loops,
                num_sampling_steps=sample_steps,
                num_diffusion_samples=1,
                calculate_confidence=True,
            )
        disto_bf = out["distogram_logits"].float()
        L_disto = disto_bf.size(1)
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)

        lm = aa_freq_loss(soft_logits, mutable_mask)

        losses = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=epi, cdr_indices=cdr,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )
        total = losses["total_loss"] + w_aa_freq * lm

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step,
            "total": float(total.item()),
            "intra": float(losses["intra_contact_loss"].item()),
            "inter": float(losses["inter_contact_loss"].item()),
            "glob": float(losses["glob_loss"].item()),
            "epi": float(losses["epitope_loss"].item()),
            "prior": float(losses["structure_prior_loss"].item()),
            "lm": float(lm.item()),
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "inter_min": diag["inter_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
            "L_disto": L_disto, "L_prior": L_prior,
        }
        history.append(record)

        if losses["inter_contact_loss"].item() < best_inter:
            best_inter = losses["inter_contact_loss"].item()
            best_seq_inter = cur_seq
            best_step_inter = step
        if losses["epitope_loss"].item() < best_epi:
            best_epi = losses["epitope_loss"].item()
            best_seq_epi = cur_seq
            best_step_epi = step
        if losses["total_loss"].item() < best_total:
            best_total = losses["total_loss"].item()
            best_seq_total = cur_seq
            best_step_total = step

        if step % snapshot_every == 0 or step == steps:
            snapshots.append({
                "step": step,
                "cdr_seq": cdr_seq,
                "full_seq": cur_seq,
                "inter": record["inter"],
                "epi": record["epi"],
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "inter_min": record["inter_min"],
                "ptm": ptm, "iptm": iptm,
            })
            with open(snapshot_path, "w") as f:
                json.dump({
                    "init_cdr": init_cdr,
                    "target_len": target_len,
                    "binder_len": binder_len,
                    "epitope": epi,
                    "framework": framework,
                    "snapshots": snapshots,
                }, f, indent=2)

        if step % log_every == 0 or step == steps:
            ptm_s = f"{ptm:.3f}" if ptm is not None else "  N/A"
            iptm_s = f"{iptm:.3f}" if iptm is not None else "  N/A"
            print(f"  {step:>4}  {record['total']:>8.3f}  "
                  f"{record['intra']:>7.3f}  {record['inter']:>7.3f}  "
                  f"{record['glob']:>7.3f}  {record['epi']:>7.3f}  "
                  f"{record['prior']:>7.3f}  "
                  f"{record['cdr_to_epi_min']:>8.2f}  {record['inter_min']:>9.2f}  "
                  f"{ptm_s:>5}  {iptm_s:>5}  {cdr_seq}")

        if step == steps:
            break

        optimizer.zero_grad()
        total.backward()
        if soft_logits.grad is None:
            print(f"  [WARN step {step}] soft_logits.grad is None — "
                  f"backprop did not reach the logits. Skipping update.")
            continue
        with torch.no_grad():
            mask = fixed_position_mask(binder_template, DEVICE)
            soft_logits.grad[mask] = 0.0
        g_norm = soft_logits.grad.norm().item()
        g_max = soft_logits.grad.abs().max().item()
        if step % log_every == 0:
            print(f"  [step {step}] grad norm={g_norm:.4f}  max={g_max:.4f}")
        optimizer.step()
        pin_fixed_positions(soft_logits, binder_template)

    dt = time.time() - t_start
    print(f"\n=== Summary ({dt/60:.1f} min) ===")
    print(f"  Initial seq CDRs: {init_cdr}")
    print(f"  Final   seq CDRs: {''.join(history[-1]['seq'][i] for i in cdr)}")
    print(f"  Best (by inter) : {''.join(best_seq_inter[i] for i in cdr)}  "
          f"(step {best_step_inter}, inter {best_inter:.4f})")
    print(f"  Best (by epi)   : {''.join(best_seq_epi[i] for i in cdr)}  "
          f"(step {best_step_epi}, epi {best_epi:.4f})")
    print(f"  Best (by total) : {''.join(best_seq_total[i] for i in cdr)}  "
          f"(step {best_step_total}, total {best_total:.4f})")
    print(f"\n  {len(snapshots)} snapshots saved to {snapshot_path}")
    return history, snapshots, (best_seq_inter, best_step_inter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--epitope-indices", required=True,
                        help="Comma-separated 0-based target residue indices")
    parser.add_argument("--framework", default="b5",
                        choices=list(__import__("test_target_pdb").INIT_FRAMEWORKS))
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    parser.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    parser.add_argument("--num-loops", type=int, default=N_LOOPS_FWD)
    parser.add_argument("--w-aa-freq", type=float, default=W_AA_FREQ)
    parser.add_argument("--wt-logit", type=float, default=3.0)
    parser.add_argument("--w-epitope", type=float, default=0.05)
    parser.add_argument("--w-intra", type=float, default=0.5)
    parser.add_argument("--w-inter", type=float, default=0.5)
    parser.add_argument("--w-glob", type=float, default=0.2)
    parser.add_argument("--w-prior", type=float, default=0.3)
    parser.add_argument("--snapshot-path", type=str,
                        default="/tmp/target_design_snapshots.json")
    parser.add_argument("--use-full-model", action="store_true",
                        help="Use Full ESMFold2 (1.3G) inside the design loop. "
                             "Slower, larger gradients may be attenuated.")
    parser.add_argument("--allow-cdr-cys", action="store_true",
                        help="Don't pin Cys to -10 in CDR positions. Required "
                             "for VHH frameworks whose H3 contains a "
                             "disulfide (e.g., KN035). Default: pin.")
    parser.add_argument("--binder-chain", default=None,
                        help="Chain ID of the binder in the PDB. If set, "
                             "the prior uses real binder CA coords for the "
                             "interface-distance constraint (v2-B5 convention). "
                             "Without this, the prior is target-only.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epi = [int(x) for x in args.epitope_indices.split(",") if x.strip()]
    run_design(
        pdb_path=args.target_pdb,
        target_chain=args.target_chain,
        epitope_indices=epi,
        framework=args.framework,
        steps=args.steps, lr=args.lr, seed=args.seed,
        log_every=args.log_every, snapshot_every=args.snapshot_every,
        sample_steps=args.sample_steps, n_loops=args.num_loops,
        w_aa_freq=args.w_aa_freq, wt_logit=args.wt_logit,
        w_epitope=args.w_epitope, w_intra=args.w_intra,
        w_inter=args.w_inter, w_glob=args.w_glob, w_prior=args.w_prior,
        snapshot_path=args.snapshot_path,
        use_full_model=args.use_full_model,
        pin_cys_in_cdr=not args.allow_cdr_cys,
        binder_chain=args.binder_chain,
    )


if __name__ == "__main__":
    main()
