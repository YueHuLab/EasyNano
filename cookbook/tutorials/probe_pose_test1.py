"""Test 1: inject rotated binder coords at diffusion start, check recovery.

The user hypothesis: ESMFold2's pose is a LOCAL minimum, not global.
If the model can recover from a perturbed start, the basin is global.
If the model gets stuck at the perturbed pose, the basin is local.

Mechanism: monkey-patch DiffusionStructureHead.sample() to use rotated coords
as the initial x (instead of pure noise). The diffusion then denoises from
this perturbed state. If the basin is global, the result converges back to
the natural pose. If local, the result stays at the rotated pose.

Note: the diffusion adds noise at each step. We add NO extra noise on top of
our injected coords (the first step's eps_std is near-zero in this schedule
because schedule[0] is the start of the noise pyramid).
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import math
import json
import types
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/huyue/esm-c-fold2")

from test_b5_pdb import setup_design  # noqa: E402
from design_b5_mps_v9_cacoord import extract_ca_per_token  # noqa: E402

DEVICE = "mps"
BINDER_LEN = 127

PRE_H1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
POST_H1_PRE_H2 = "SLGWFRQAPGQGLEAVAAI"
POST_H2_PRE_H3 = "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA"
POST_H3 = "WGQGTLVTVS"

CDR = "GLQIGYGMYMSYSGQKRVVTDSSTPIYKAGIR"  # v9_best_15seed
BINDER = (PRE_H1 + CDR[:10] + POST_H1_PRE_H2 + CDR[10:16]
          + POST_H2_PRE_H3 + CDR[16:] + POST_H3)
assert len(BINDER) == 127


def load_model_full():
    print("Loading FULL ESMFold2 (1.3G) ...", flush=True)
    t0 = time.time()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    config = ESMFold2Config.from_pretrained("/Users/huyue/esm-c-fold2/ESMFold2")
    config.esmc_id = "/Users/huyue/esm-c-fold2/ESMC-6B"
    model = ESMFold2Model.from_pretrained(
        "/Users/huyue/esm-c-fold2/ESMFold2", config=config
    ).float().to(DEVICE).eval()
    from esmscore.score_only import _patch_for_mps
    _patch_for_mps(model)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model


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


def rotation_matrix_y(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# --- Custom sample method that supports x_init ---
import transformers.models.esmfold2.modeling_esmfold2_common as _esmfold2_mod
_DiffusionStructureHead = _esmfold2_mod.DiffusionStructureHead
_original_sample = _DiffusionStructureHead.sample


def _patched_sample(self, z_trunk, s_inputs, s_trunk,
                    relative_position_encoding, ref_pos, ref_charge, ref_mask,
                    ref_element, ref_atom_name_chars, ref_space_uid, tok_idx,
                    asym_id, residue_index, entity_id, token_index, sym_id,
                    token_attention_mask=None, num_diffusion_samples=1,
                    num_sampling_steps=None, max_inference_sigma=256.0,
                    noise_scale=None, step_scale=None,
                    return_atom_repr=False, use_inference_cache=True,
                    denoising_early_exit_rmsd=None, x_init=None):
    """Same as original sample, but uses x_init as initial x (or random)."""
    n_atoms = tok_idx.shape[1]
    device = s_inputs.device
    target_batch = s_inputs.shape[0] * num_diffusion_samples

    inference_cache = {} if use_inference_cache else None

    steps = self.inference_num_steps if num_sampling_steps is None \
            else int(num_sampling_steps)
    schedule = self.inference_noise_schedule(steps, device)
    if max_inference_sigma is not None:
        schedule = schedule[schedule <= float(max_inference_sigma)]
        schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

    lam = self.noise_scale if noise_scale is None else float(noise_scale)
    eta = self.step_scale if step_scale is None else float(step_scale)

    if x_init is not None:
        x = x_init.float().to(device).expand(target_batch, -1, -1).contiguous().clone()
    else:
        x = schedule[0] * torch.randn(
            target_batch, n_atoms, 3, device=device, dtype=torch.float32
        )
    atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

    gammas = torch.where(
        schedule > self.gamma_min,
        torch.full_like(schedule, self.gamma_0),
        torch.zeros_like(schedule),
    )

    x_denoised_prev = None
    token_repr = None
    diff_atom_intermediates = None

    step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
    num_steps = len(step_pairs)

    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(step_pairs):
        x, x_denoised_prev = self._center_random_augmentation(
            x, atom_mask, second_coords=x_denoised_prev
        )

        sigma_tm_val = float(sigma_tm.item())
        t_hat_val = sigma_tm_val * (1.0 + float(gamma.item()))
        eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
        x_noisy = x + eps_std * torch.randn_like(x)

        is_last_step = step_idx == num_steps - 1
        request_atom_repr = return_atom_repr and (
            is_last_step or denoising_early_exit_rmsd is not None
        )

        dm_out = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.full(
                (target_batch,), t_hat_val, device=device, dtype=torch.float32
            ),
            ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask,
            ref_element=ref_element, ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid, tok_idx=tok_idx,
            s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            asym_id=asym_id, residue_index=residue_index,
            entity_id=entity_id, token_index=token_index, sym_id=sym_id,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=True, return_atom_repr=request_atom_repr,
            inference_cache=inference_cache,
        )

        x_denoised = dm_out["x_denoised"]
        token_repr = dm_out["token_repr"]
        if request_atom_repr:
            diff_atom_intermediates = dm_out.get("atom_intermediates")

        with torch.autocast(device_type="cuda", enabled=False):
            x_noisy = self._weighted_rigid_align(
                x_noisy.float(), x_denoised.float(), atom_mask, atom_mask
            )
        x_noisy = x_noisy.to(dtype=x_denoised.dtype)

        sigma_t_val = float(sigma_t.item())
        denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
        x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

        if (denoising_early_exit_rmsd is not None and x_denoised_prev is not None
                and step_idx >= 1):
            with torch.autocast(device_type="cuda", enabled=False):
                aligned = self._weighted_rigid_align(
                    x_denoised_prev.float(), x_denoised.float(),
                    atom_mask, atom_mask,
                )
            diff = (x_denoised.float() - aligned) * atom_mask.unsqueeze(-1)
            per_sample_rmsd = (
                diff.pow(2).sum(dim=(-1, -2)) / atom_mask.sum(dim=-1).clamp(min=1)
            ).sqrt()
            if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                x = x_denoised
                x_denoised_prev = x_denoised
                break

        x_denoised_prev = x_denoised

    return {
        "sample_atom_coords": x,
        "diff_token_repr": token_repr,
        "diff_atom_intermediates": diff_atom_intermediates,
    }


def make_patched_sample(x_init):
    """Return a function that wraps _patched_sample with x_init baked in."""
    def _fn(self, *args, **kwargs):
        return _patched_sample(self, *args, x_init=x_init, **kwargs)
    return _fn


def main():
    setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
    target_seq = setup["target_sequence"]
    epi = list(setup["epitope_token_indices"])
    cdr = list(setup["cdr_indices"])
    print(f"target_len={len(target_seq)} binder_len={BINDER_LEN}")
    print(f"epitope (21): {epi}")
    print(f"CDR: {CDR}")

    out_dir = Path("/tmp/b5_pose_test1")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()

    model = load_model_full()

    # Build features
    from esmscore._complex import build_complex_features
    feats = build_complex_features(BINDER, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith("_")}
    features = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    atom_to_token = features["atom_to_token"][0]
    ref_atom_name_chars = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0]
    n_real_atoms = int(atom_mask.sum().item())
    print(f"  n_real_atoms = {n_real_atoms}", flush=True)

    # asym_id is per-token; atom_to_token maps atoms → tokens.
    # Token 0..BINDER_LEN-1 = binder (asym=0)
    # Token BINDER_LEN..L-1 = target (asym=1)
    asym_id_tokens = features["asym_id"][0]  # [L]
    atom_asym = asym_id_tokens[atom_to_token]  # [N_atoms]
    binder_atom_mask = (atom_asym == 0)  # [N_atoms]
    target_atom_mask = (atom_asym == 1)  # [N_atoms]
    n_binder_atoms = int(binder_atom_mask.sum().item())
    n_target_atoms = int(target_atom_mask.sum().item())
    print(f"  n_binder_atoms = {n_binder_atoms}  "
          f"n_target_atoms = {n_target_atoms}", flush=True)

    # === Step 0: Baseline fold (no injection) ===
    print(f"\n=== Step 0: Baseline fold (no injection) ===", flush=True)
    SEED = 11
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    with torch.inference_mode():
        out0 = model.forward(
            **features,
            num_loops=3, num_sampling_steps=14,
            num_diffusion_samples=1, calculate_confidence=True,
        )
    dt0 = time.time() - t0
    iptm0 = float(out0["iptm"][0].item())
    ptm0 = float(out0["ptm"][0].item())
    pae0 = out0["pae"][0].float().cpu().numpy()
    print(f"  iptm={iptm0:.3f}  pTM={ptm0:.3f}  ({dt0:.1f}s)", flush=True)

    sample_coords0 = out0["sample_atom_coords"].float()
    if sample_coords0.dim() == 4:
        sample_coords0 = sample_coords0[:, 0]
    if sample_coords0.dim() == 2:
        sample_coords0 = sample_coords0.unsqueeze(0)
    sample_coords0 = sample_coords0[0]  # [N_atoms, 3]
    ca0 = extract_ca_per_token(
        sample_coords0.unsqueeze(0), atom_to_token,
        ref_atom_name_chars, atom_mask
    )[0].cpu().numpy()
    np.save(out_dir / "baseline_binder_ca.npy", ca0[:BINDER_LEN])
    np.save(out_dir / "baseline_target_ca.npy", ca0[BINDER_LEN:])
    np.save(out_dir / "baseline_sample_coords.npy", sample_coords0.cpu().numpy())
    print(f"  baseline sample_coords shape: {sample_coords0.shape}",
          flush=True)

    # --- Build rotated-x_init for binder atoms ---
    # We rotate the binder atom coords 30° around y-axis, through binder centroid
    binder_coords = sample_coords0[binder_atom_mask].cpu().numpy()  # [N_b, 3]
    binder_centroid = binder_coords.mean(0)
    R = rotation_matrix_y(math.radians(30.0))
    rotated = (binder_coords - binder_centroid) @ R.T + binder_centroid
    print(f"\n  Baseline binder centroid: {binder_centroid}", flush=True)
    print(f"  Rotation: 30° around y-axis through centroid", flush=True)
    print(f"  Rotated binder CA distance from baseline: "
          f"{np.linalg.norm(rotated.mean(0) - binder_coords.mean(0)):.2f}Å",
          flush=True)

    # Compute CA-only rotation for inspection
    binder_ca = ca0[:BINDER_LEN]
    binder_ca_centroid = binder_ca.mean(0)
    rotated_ca = (binder_ca - binder_ca_centroid) @ R.T + binder_ca_centroid
    ca_shift = float(np.sqrt(((rotated_ca - binder_ca) ** 2).sum(-1).mean()))
    ca_max_shift = float(np.sqrt(((rotated_ca - binder_ca) ** 2).sum(-1).max()))
    print(f"  Rotated binder CAs: mean shift={ca_shift:.2f}Å  "
          f"max shift={ca_max_shift:.2f}Å", flush=True)

    # Build the x_init tensor: same as baseline coords, but binder atoms rotated
    x_init = sample_coords0.clone()
    x_init[binder_atom_mask] = torch.from_numpy(rotated).to(
        dtype=x_init.dtype, device=x_init.device
    )

    rotated_x_init_ca = extract_ca_per_token(
        x_init.unsqueeze(0), atom_to_token,
        ref_atom_name_chars, atom_mask
    )[0].cpu().numpy()

    # === Step 1: Test that ref_pos is NOT the injection point ===
    # (We expect modifying ref_pos to do nothing because it's per-residue local frame.)
    # Skip this — we already know ref_pos is local-frame.

    # === Step 2: Fold with x_init = rotated binder (no noise on top) ===
    print(f"\n=== Step 2: Fold with rotated-binder x_init (seed={SEED}) ===",
          flush=True)
    # Monkey-patch
    model.structure_head.sample = types.MethodType(
        make_patched_sample(x_init), model.structure_head
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t1 = time.time()
    try:
        with torch.inference_mode():
            out1 = model.forward(
                **features,
                num_loops=3, num_sampling_steps=14,
                num_diffusion_samples=1, calculate_confidence=True,
            )
    finally:
        # Restore original
        model.structure_head.sample = _original_sample
    dt1 = time.time() - t1
    iptm1 = float(out1["iptm"][0].item())
    ptm1 = float(out1["ptm"][0].item())
    pae1 = out1["pae"][0].float().cpu().numpy()
    print(f"  iptm={iptm1:.3f}  pTM={ptm1:.3f}  ({dt1:.1f}s)", flush=True)

    sample_coords1 = out1["sample_atom_coords"].float()
    if sample_coords1.dim() == 4:
        sample_coords1 = sample_coords1[:, 0]
    if sample_coords1.dim() == 2:
        sample_coords1 = sample_coords1.unsqueeze(0)
    sample_coords1 = sample_coords1[0]
    ca1 = extract_ca_per_token(
        sample_coords1.unsqueeze(0), atom_to_token,
        ref_atom_name_chars, atom_mask
    )[0].cpu().numpy()
    np.save(out_dir / "rotated_binder_ca.npy", ca1[:BINDER_LEN])
    np.save(out_dir / "rotated_target_ca.npy", ca1[BINDER_LEN:])
    np.save(out_dir / "rotated_sample_coords.npy", sample_coords1.cpu().numpy())

    # === Step 3: Fold with x_init = baseline coords (no rotation) - control ===
    # If the model uses x_init, this should reproduce the baseline.
    # If the model ignores x_init, this should be ~noise-driven and differ.
    print(f"\n=== Step 3: Control - fold with x_init = BASELINE coords (no rotation) ===",
          flush=True)
    x_init_baseline = sample_coords0.clone()
    model.structure_head.sample = types.MethodType(
        make_patched_sample(x_init_baseline), model.structure_head
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t2 = time.time()
    try:
        with torch.inference_mode():
            out2 = model.forward(
                **features,
                num_loops=3, num_sampling_steps=14,
                num_diffusion_samples=1, calculate_confidence=True,
            )
    finally:
        model.structure_head.sample = _original_sample
    dt2 = time.time() - t2
    iptm2 = float(out2["iptm"][0].item())
    ptm2 = float(out2["ptm"][0].item())
    print(f"  iptm={iptm2:.3f}  pTM={ptm2:.3f}  ({dt2:.1f}s)", flush=True)
    sample_coords2 = out2["sample_atom_coords"].float()
    if sample_coords2.dim() == 4:
        sample_coords2 = sample_coords2[:, 0]
    if sample_coords2.dim() == 2:
        sample_coords2 = sample_coords2.unsqueeze(0)
    sample_coords2 = sample_coords2[0]
    ca2 = extract_ca_per_token(
        sample_coords2.unsqueeze(0), atom_to_token,
        ref_atom_name_chars, atom_mask
    )[0].cpu().numpy()
    np.save(out_dir / "control_binder_ca.npy", ca2[:BINDER_LEN])
    np.save(out_dir / "control_target_ca.npy", ca2[BINDER_LEN:])

    # === Analysis ===
    print(f"\n{'='*60}", flush=True)
    print(f"=== ANALYSIS: Did x_init injection actually take effect? ===",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    b_base = ca0[:BINDER_LEN]
    t_base = ca0[BINDER_LEN:]
    b_rot = ca1[:BINDER_LEN]
    t_rot = ca1[BINDER_LEN:]
    b_ctrl = ca2[:BINDER_LEN]
    t_ctrl = ca2[BINDER_LEN:]

    # Method: Kabsch align on target, compare binder
    def align_on_target(bi, ti, bj, tj):
        offset = tj.mean(0) - ti.mean(0)
        return bi + offset, kabsch_rmsd(bi + offset, bj)

    _, rmsd_base_rot = align_on_target(b_rot, t_rot, b_base, t_base)
    _, rmsd_base_ctrl = align_on_target(b_ctrl, t_ctrl, b_base, t_base)
    _, rmsd_rot_ctrl = align_on_target(b_ctrl, t_ctrl, b_rot, t_rot)
    # Self-comparisons
    _, rmsd_base_self = align_on_target(b_base, t_base, b_base, t_base)

    # Also: pure x_init alignment — what would the result be if x_init was used as-is?
    # This requires extracting binder CAs from the rotated x_init
    rotated_x_init_ca = extract_ca_per_token(
        x_init.unsqueeze(0), atom_to_token,
        ref_atom_name_chars, atom_mask
    )[0].cpu().numpy()
    x_init_binder_ca = rotated_x_init_ca[:BINDER_LEN]
    x_init_target_ca = rotated_x_init_ca[BINDER_LEN:]
    _, rmsd_xinit_base = align_on_target(
        x_init_binder_ca, x_init_target_ca, b_base, t_base
    )
    _, rmsd_xinit_rot = align_on_target(
        x_init_binder_ca, x_init_target_ca, b_rot, t_rot
    )

    print(f"  Binder CA RMSD (target-aligned):", flush=True)
    print(f"    baseline (self):                 {rmsd_base_self:.2f}Å",
          flush=True)
    print(f"    baseline ↔ rotated-fold:         {rmsd_base_rot:.2f}Å",
          flush=True)
    print(f"    baseline ↔ control-fold:         {rmsd_base_ctrl:.2f}Å",
          flush=True)
    print(f"    rotated-fold ↔ control-fold:     {rmsd_rot_ctrl:.2f}Å",
          flush=True)
    print(f"    x_init (injected) ↔ baseline:    {rmsd_xinit_base:.2f}Å",
          flush=True)
    print(f"    x_init (injected) ↔ rotated-fold:{rmsd_xinit_rot:.2f}Å",
          flush=True)

    # Interface centroid
    def iface_centroid_dist(b_ca, t_ca):
        bc = b_ca[cdr].mean(0)
        tc = t_ca[epi].mean(0)
        return float(np.linalg.norm(bc - tc))

    print(f"\n  CDR-epitope centroid distance:", flush=True)
    print(f"    baseline:        {iface_centroid_dist(b_base, t_base):.2f}Å",
          flush=True)
    print(f"    rotated-fold:    {iface_centroid_dist(b_rot, t_rot):.2f}Å",
          flush=True)
    print(f"    control-fold:    {iface_centroid_dist(b_ctrl, t_ctrl):.2f}Å",
          flush=True)

    # Save
    summary = {
        "binder": BINDER, "cdr": CDR, "seed": SEED,
        "rotation_deg": 30.0, "axis": "y",
        "iptm_baseline": iptm0, "iptm_rotated": iptm1, "iptm_control": iptm2,
        "ptm_baseline": ptm0, "ptm_rotated": ptm1, "ptm_control": ptm2,
        "rmsd_base_rot": rmsd_base_rot,
        "rmsd_base_ctrl": rmsd_base_ctrl,
        "rmsd_rot_ctrl": rmsd_rot_ctrl,
        "rmsd_xinit_base": rmsd_xinit_base,
        "rmsd_xinit_rot": rmsd_xinit_rot,
        "iface_baseline": iface_centroid_dist(b_base, t_base),
        "iface_rotated": iface_centroid_dist(b_rot, t_rot),
        "iface_control": iface_centroid_dist(b_ctrl, t_ctrl),
    }
    with open(out_dir / "test1.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Verdict
    print(f"\n  Verdict:", flush=True)
    print(f"  ----------------------------------------", flush=True)
    if rmsd_xinit_base < 0.5:
        print(f"  x_init injection is essentially ignored (injected pose "
              f"was {rmsd_xinit_base:.1f}Å from baseline, but model "
              f"produced pose {rmsd_base_rot:.1f}Å from baseline).",
              flush=True)
        if rmsd_base_rot < 2.0:
            print(f"  → Despite x_init = rotated binder, fold recovered to "
                  f"baseline ({rmsd_base_rot:.2f}Å RMSD). "
                  f"POSE BASIN IS GLOBAL (or at least very strong).",
                  flush=True)
        else:
            print(f"  → x_init = rotated binder, fold ended up at "
                  f"{rmsd_base_rot:.2f}Å from baseline. "
                  f"Local minimum IS sticky.",
                  flush=True)
    else:
        if rmsd_base_rot < 1.0:
            print(f"  x_init had only minor effect. Fold recovered to "
                  f"baseline ({rmsd_base_rot:.2f}Å). The basin is very strong.",
                  flush=True)
        else:
            print(f"  Mixed signal. Inspect JSON for details.", flush=True)

    print(f"\nResults saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
