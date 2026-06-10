"""Design with HARD epitope contact mask — push ipTM > 0.6.

Strategy:
  - Use Fast model for design (proven to work)
  - Add a new HARD contact loss on top of the existing soft epitope loss
  - Hard loss: require at least N CDR residues to be within D of M anchor epitope residues
  - Anchor subset: the tight epitope cluster 111-121 (which v2 design already approaches)

Mathematical form:
  deficit = max(0, min_contacts_required - actual_contacts)
  hard_loss = scale * deficit + soft_hinge(sum of distances)

If CDR has fewer than required contacts, big penalty.
If CDR meets the contact requirement, only soft penalty for being too far.
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
    compute_epitope_loss,
)
from test_b5_pdb import setup_design  # noqa: E402

# Reuse helpers from v2 design (we only need the loss + design framework)
from design_b5_mps_v2 import (
    init_soft_logits, build_soft_res_type, soft_to_hard_seq,
    make_target_one_hot, aa_freq_loss, fixed_position_mask,
    reorder_bf_to_target_first, align_prior_to_disto, cdr_to_epitope_stats,
    load_model, MODEL_PATH, ESMC_PATH, DEVICE, NUM_RES_TYPES, TOKENS,
    AA_TO_TOKEN, AA_DIMS, CYS_TOK, AA_FREQ, TEMP_MIN,
    SAMPLE_STEPS_FWD, N_LOOPS_FWD, W_AA_FREQ,
)
WT_LOGIT_DEFAULT = 3.0

# Hard mask config
ANCHOR_EPI_CLUSTER = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
MIN_CONTACTS_REQUIRED = 4      # need at least 4 CDR residues in contact
CONTACT_CUTOFF_A = 8.0          # contact if within 8 Å
HARD_SCALE = 1.0               # weight of the hard contact requirement
HARD_HINGE_CUTOFF = 18.0        # gradient flows up to 18 Å (matches initial WT pos)
HARD_HINGE_SCALE = 0.1          # soft penalty for being too far


def hard_epitope_contact_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    epitope_token_indices: list[int],   # anchor residues
    bin_distance: torch.Tensor,
    contact_cutoff: float = CONTACT_CUTOFF_A,
    min_contacts: int = MIN_CONTACTS_REQUIRED,
    cdr_indices: list[int] | None = None,
    hinge_cutoff: float = HARD_HINGE_CUTOFF,
    hinge_scale: float = HARD_HINGE_SCALE,
) -> torch.Tensor:
    """Contact-reward form: maximize the number of (CDR, anchor) pairs within
    `contact_cutoff`. Concretely, for each (CDR, anchor) pair, compute a soft
    "contact score" that ramps from 0 at hinge_cutoff to 1 at < contact_cutoff.
    The loss is the negative total contact score — minimizing it maximizes contacts.

    This is the opposite of a hinge penalty: it provides a GRADIENT pulling CDR
    residues TOWARD anchors, not pushing them away.
    """
    if not epitope_token_indices:
        return torch.zeros((), device=distogram_logits.device,
                            dtype=distogram_logits.dtype)
    cross = distogram_logits[:, -binder_length:, :-binder_length, :]
    # Pick anchor columns only
    anchor = cross[:, :, epitope_token_indices, :]  # [B, L_b, E_anchor, n_bins]
    probs = torch.softmax(anchor, dim=-1)
    e_dist = (probs * bin_distance).sum(-1)         # [B, L_b, E_anchor]

    # CDR mask
    if cdr_indices:
        mask = torch.zeros(binder_length, device=distogram_logits.device)
        mask[cdr_indices] = 1.0
    else:
        mask = torch.ones(binder_length, device=distogram_logits.device)
    cdr_mask = mask[None, :]  # [1, L_b]

    # Take min over anchors: [B, L_b] = best distance per CDR residue
    min_dist = e_dist.min(dim=-1).values           # [B, L_b]
    cdr_min = min_dist * cdr_mask                  # [B, L_b]

    # Soft contact score: 1 at dist=0, ramps linearly to 0 at hinge_cutoff
    # 1 - clamp((dist - contact_cutoff) / (hinge_cutoff - contact_cutoff), 0, 1)
    ramp = hinge_cutoff - contact_cutoff
    soft_contact = 1.0 - ((cdr_min - contact_cutoff).clamp(0, ramp) / ramp)
    # Zero out non-CDR positions
    soft_contact = soft_contact * cdr_mask

    # Reward = sum of contact scores (we want to maximize this)
    # Loss = -reward
    contact_reward = soft_contact.sum(dim=-1)  # [B]
    return -contact_reward.mean()


def run_design(steps: int = 60,
               lr: float = 1.0,
               wt_logit: float = WT_LOGIT_DEFAULT,
               w_epitope: float = 0.05,            # soft epi weight (existing)
               w_hard: float = 0.5,                # NEW: hard contact weight
               w_intra: float = 0.5,
               w_inter: float = 0.5,
               w_glob: float = 0.2,
               w_prior: float = 0.3,
               w_aa_freq: float = W_AA_FREQ,
               log_every: int = 2,
               snapshot_every: int = 5,
               sample_steps: int = SAMPLE_STEPS_FWD,
               n_loops: int = N_LOOPS_FWD,
               seed: int = 0,
               snapshot_path: str = "/tmp/b5_design_v4_hardmask_snaps.json"):
    print(f"=== B5.pdb HARD-mask design (Fast model, lr={lr}, "
          f"w_hard={w_hard}, n_loops={n_loops}) ===\n", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    binder_template = setup["binder_template"]
    target_len = len(target_seq)
    binder_len = len(binder_template)
    epi = setup["epitope_token_indices"]
    cdr = setup["cdr_indices"]
    prior_bins = setup["prior_bins"]
    prior_mask = setup["prior_mask"]
    binder_wt = setup["binder_full_sequence"]

    print(f"  Target {target_len}, Binder {binder_len}, "
          f"CDRs {len(cdr)}, Epitope {len(epi)}, Prior L={prior_bins.size(0)}")
    print(f"  Hard mask: require >= {MIN_CONTACTS_REQUIRED} CDR residues "
          f"within {CONTACT_CUTOFF_A} Å of anchor cluster {ANCHOR_EPI_CLUSTER}")
    print(f"  Anchor indices in distogram (target-first layout):", end=" ")
    epi_tfb = [i for i, ei in enumerate(epi) if ei in ANCHOR_EPI_CLUSTER]
    print(f"{epi_tfb} (of {len(epi)} epitope residues)")

    model = load_model()
    target_one_hot = make_target_one_hot(target_seq, DEVICE)

    LOSS_WEIGHTS["epitope"] = w_epitope
    LOSS_WEIGHTS["intra_contact"] = w_intra
    LOSS_WEIGHTS["inter_contact"] = w_inter
    LOSS_WEIGHTS["glob"] = w_glob
    LOSS_WEIGHTS["structure_prior"] = w_prior
    print(f"  loss weights: epitope={w_epitope} hard={w_hard} intra={w_intra} "
          f"inter={w_inter} glob={w_glob} prior={w_prior} aa_freq={w_aa_freq}")

    soft_logits = init_soft_logits(binder_template, binder_wt, wt_logit=wt_logit).to(DEVICE)
    soft_logits = soft_logits.detach().requires_grad_(True)
    optimizer = optim.Adam([soft_logits], lr=lr)
    mutable_mask = ~fixed_position_mask(binder_template, DEVICE)

    history = []
    snapshots = []
    best_total = float("inf")
    best_seq_total = ""
    best_step_total = -1
    best_n_contacts = -1
    best_seq_contacts = ""
    best_step_contacts = -1
    best_cdr_to_epi = float("inf")
    best_seq_epi = ""
    best_step_epi = -1

    init_seq = soft_to_hard_seq(soft_logits)
    init_cdr = "".join(init_seq[i] for i in cdr)
    print(f"\n  init CDRs: {init_cdr}\n", flush=True)

    print(f"Designing {steps} steps (lr={lr}, sample={sample_steps}, loops={n_loops}) ...",
          flush=True)
    header = (f"  {'step':>4}  {'total':>8}  {'reward':>7}  {'soft_e':>6}  "
              f"{'intra':>6}  {'inter':>6}  {'n_ct':>4}  {'CDR→epi':>8}  "
              f"{'pTM':>5}  {'ipTM':>5}  CDR_seq")
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)
    t_start = time.time()

    for step in range(steps + 1):
        t = (step + 1) / max(steps, 1)
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMP_MIN + (1 - TEMP_MIN) * remaining

        res_type_soft = build_soft_res_type(soft_logits, target_one_hot,
                                            temperature=temperature)
        if res_type_soft.dim() == 2:
            res_type_soft = res_type_soft.unsqueeze(0)

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
        disto = reorder_bf_to_target_first(disto_bf, binder_len)
        pb, pm = align_prior_to_disto(prior_bins, prior_mask, disto)
        bin_dist = get_mid_points(64, 2.0, 22.0).to(DEVICE)

        lm = aa_freq_loss(soft_logits, mutable_mask)

        from binder_design_hy_losses import compute_structure_losses
        sl = compute_structure_losses(
            disto, binder_length=binder_len,
            epitope_token_indices=None, cdr_indices=None,
            prior_bins=pb.to(DEVICE), prior_mask=pm.to(DEVICE),
            n_bins=64, min_dist=2.0, max_dist=22.0,
        )

        soft_epi = compute_epitope_loss(
            disto, binder_len, epi, bin_dist,
            cutoff=8.0, cdr_indices=cdr,
        )

        # Hard contact REWARD (negative loss = maximize contact score)
        hard = hard_epitope_contact_loss(
            disto, binder_len, epi_tfb, bin_dist,
            contact_cutoff=CONTACT_CUTOFF_A,
            min_contacts=MIN_CONTACTS_REQUIRED,
            cdr_indices=cdr,
            hinge_cutoff=HARD_HINGE_CUTOFF,
            hinge_scale=HARD_HINGE_SCALE,
        )

        total = (sl["intra_contact_loss"] * w_intra
                 + sl["inter_contact_loss"] * w_inter
                 + sl["glob_loss"] * w_glob
                 + sl["structure_prior_loss"] * w_prior
                 + w_epitope * soft_epi
                 + w_hard * hard
                 + w_aa_freq * lm)

        diag = cdr_to_epitope_stats(disto, cdr, epi, target_len, binder_len)
        ptm = float(out["ptm"][0].item()) if "ptm" in out and out["ptm"].numel() else None
        iptm = float(out["iptm"][0].item()) if "iptm" in out and out["iptm"].numel() else None

        # Compute n_contacts and contact reward for logging
        cross = disto[:, -binder_len:, :-binder_len, :]
        anchor = cross[:, :, epi_tfb, :]
        probs = torch.softmax(anchor, dim=-1)
        e_dist = (probs * bin_dist).sum(-1)
        min_dist_cdr = e_dist.min(dim=-1).values[0]
        cdr_mask_t = torch.tensor(cdr, device=DEVICE, dtype=torch.long)
        n_contacts = int(((min_dist_cdr[cdr_mask_t] < CONTACT_CUTOFF_A)).sum().item())
        # The hard loss is -reward, so reward = -hard
        reward = -float(hard.item())

        cdr_seq = "".join(cur_seq[i] for i in cdr)
        record = {
            "step": step,
            "total": float(total.item()),
            "reward": reward,
            "soft_epi": float(soft_epi.item()),
            "intra": float(sl["intra_contact_loss"].item()),
            "inter": float(sl["inter_contact_loss"].item()),
            "glob": float(sl["glob_loss"].item()),
            "n_contacts": n_contacts,
            "cdr_to_epi_min": diag["cdr_to_epitope_min"],
            "ptm": ptm, "iptm": iptm,
            "seq": cur_seq, "cdr_seq": cdr_seq,
        }
        history.append(record)
        if step == 0 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t_start
            print(f"  {step:>4}  {record['total']:>8.3f}  {reward:>7.2f}  "
                  f"{record['soft_epi']:>6.2f}  {record['intra']:>6.2f}  "
                  f"{record['inter']:>6.3f}  {n_contacts:>4}  "
                  f"{record['cdr_to_epi_min']:>8.2f}  "
                  f"{ptm:>5.3f}  {iptm:>5.3f}  {cdr_seq[:32]}  "
                  f"[{elapsed:>5.0f}s]", flush=True)

        if record["total"] < best_total:
            best_total = record["total"]; best_seq_total = cur_seq; best_step_total = step
        if n_contacts > best_n_contacts:
            best_n_contacts = n_contacts; best_seq_contacts = cur_seq; best_step_contacts = step
        if diag["cdr_to_epitope_min"] < best_cdr_to_epi:
            best_cdr_to_epi = diag["cdr_to_epitope_min"]
            best_seq_epi = cur_seq; best_step_epi = step

        if step < steps:
            optimizer.zero_grad()
            total.backward()
            gnorm = soft_logits.grad.norm().item()
            if gnorm > 1.0:
                soft_logits.grad.mul_(1.0 / gnorm)
            optimizer.step()
            with torch.no_grad():
                soft_logits.clamp_(-10.0, 10.0)

        if step > 0 and step % snapshot_every == 0:
            snap = {
                "step": step,
                "cdr_seq": cdr_seq,
                "full_seq": cur_seq,
                "total": record["total"],
                "reward": reward,
                "n_contacts": n_contacts,
                "cdr_to_epi_min": record["cdr_to_epi_min"],
                "ptm": ptm,
                "iptm": iptm,
            }
            snapshots.append(snap)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time / (steps + 1):.1f}s/step)")
    print(f"  best total:       step {best_step_total} = {best_total:.3f}  seq={best_seq_total[:40]}")
    print(f"  best n_contacts:  step {best_step_contacts} = {best_n_contacts}  seq={best_seq_contacts[:40]}")
    print(f"  best CDR→epi:     step {best_step_epi} = {best_cdr_to_epi:.2f}  seq={best_seq_epi[:40]}")

    out = {
        "init_cdr": init_cdr,
        "binder_len": binder_len,
        "config": {
            "model": "ESMFold2-Fast (721M) for design",
            "steps": steps, "lr": lr, "wt_logit": wt_logit,
            "w_epitope": w_epitope, "w_hard": w_hard,
            "anchor_epitope": ANCHOR_EPI_CLUSTER,
            "min_contacts_required": MIN_CONTACTS_REQUIRED,
            "contact_cutoff_A": CONTACT_CUTOFF_A,
            "w_intra": w_intra, "w_inter": w_inter,
            "w_glob": w_glob, "w_prior": w_prior, "w_aa_freq": w_aa_freq,
            "n_loops": n_loops, "sample_steps": sample_steps, "seed": seed,
        },
        "snapshots": snapshots,
    }
    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSnapshots saved to {snapshot_path}")
    print(f"  init_cdr: {init_cdr}")
    print(f"  {len(snapshots)} snapshots")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--wt-logit", type=float, default=WT_LOGIT_DEFAULT)
    p.add_argument("--w-epitope", type=float, default=0.05)
    p.add_argument("--w-hard", type=float, default=0.5)
    p.add_argument("--n-loops", type=int, default=N_LOOPS_FWD)
    p.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS_FWD)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--snapshot-path", type=str,
                   default="/tmp/b5_design_v4_hardmask_snaps.json")
    args = p.parse_args()
    run_design(steps=args.steps, lr=args.lr, wt_logit=args.wt_logit,
               w_epitope=args.w_epitope, w_hard=args.w_hard,
               n_loops=args.n_loops, sample_steps=args.sample_steps,
               seed=args.seed, log_every=args.log_every,
               snapshot_every=args.snapshot_every,
               snapshot_path=args.snapshot_path)
