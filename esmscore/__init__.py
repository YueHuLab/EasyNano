"""
ESMscore: Fast protein confidence scoring — skip diffusion, score only.

    scorer = ESMScorer(model_path="/path/to/ESMFold2-Fast",
                       esmc_path="/path/to/ESMC-6B")
    result = scorer.score("MQIFVKTLTG...")
    # -> {"ptm": 0.85, "iptm": None, "pae": [...]}

Two modes:
  - "score_only": Skip diffusion entirely, pTM/ipTM/PAE from pair rep (fastest)
  - "fast": num_sampling_steps=1, get full pLDDT + pTM + ipTM + PAE
"""

from __future__ import annotations

import numpy as np
import torch


class ESMScorer:
    """Confidence scoring using ESMFold2. Default: score_only (no diffusion)."""

    def __init__(
        self,
        model_path: str | None = None,
        esmc_path: str | None = None,
        device: str | None = None,
        mode: str = "score_only",
        pdb_path: str | None = None,
        chain_a_len: int | None = None,
        chain_b_len: int | None = None,
    ):
        self.model_path = model_path
        self.esmc_path = esmc_path
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.mode = mode
        self.pdb_path = pdb_path
        self.chain_a_len = chain_a_len
        self.chain_b_len = chain_b_len
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        mp = self.model_path
        if mp is None:
            # Try common local paths
            import os
            for cand in [
                os.path.expanduser("~/esm-c-fold2/ESMFold2-Fast"),
                "/Users/huyue/esm-c-fold2/ESMFold2-Fast",
            ]:
                if os.path.isdir(cand):
                    mp = cand
                    break
            if mp is None:
                mp = "biohub/ESMFold2-Fast"

        config = ESMFold2Config.from_pretrained(mp)

        # Override esmc_id to local path if available
        if self.esmc_path:
            config.esmc_id = self.esmc_path
        else:
            import os
            for cand in [
                os.path.expanduser("~/esm-c-fold2/ESMC-6B"),
                "/Users/huyue/esm-c-fold2/ESMC-6B",
            ]:
                if os.path.isdir(cand):
                    config.esmc_id = cand
                    break

        # Device selection: MPS > CUDA > CPU
        if self.device == "mps" or (self.device == "auto" and torch.backends.mps.is_available()):
            self._model = ESMFold2Model.from_pretrained(mp, config=config).float().to("mps").eval()
            from .score_only import _patch_for_mps
            _patch_for_mps(self._model)
        elif self.device == "cuda":
            self._model = ESMFold2Model.from_pretrained(mp, config=config).eval().to("cuda")
        else:
            self._model = ESMFold2Model.from_pretrained(mp, config=config).eval().float().cpu()

    def score(self, sequence: str, num_loops: int = 0, **kwargs) -> dict:
        """Score a single protein sequence. Returns pTM, ipTM, PAE."""
        self._ensure_model()

        if self.mode == "score_only":
            return self._score_only(sequence, num_loops=num_loops)
        else:
            return self._score_with_diffusion(sequence, num_loops=num_loops, **kwargs)

    def _score_only(self, sequence: str, num_loops: int = 1) -> dict:
        """Skip diffusion entirely."""
        from transformers.models.esmfold2.protein_utils import prepare_protein_features
        from .score_only import score_only_forward

        # prepare_protein_features returns already-batched tensors [B, ...]
        features = prepare_protein_features(sequence)

        result = score_only_forward(self._model, features, num_loops=num_loops)
        return {
            "ptm": float(result["ptm"][0].item()),
            "iptm": float(result["iptm"][0].item()) if result["iptm"].numel() else None,
            "pae": result["pae"][0].cpu().numpy(),
        }

    # ---- Complex (multi-chain) scoring ----

    @staticmethod
    def _prepare_complex_features(seq_a: str, seq_b: str) -> dict:
        """Build features for a two-chain complex (A+B) with proper chain separation."""
        from ._complex import build_complex_features
        return build_complex_features(seq_a, seq_b)

    def _get_pdb_coords(self, binder: str, target: str) -> torch.Tensor | None:
        """Build PDB coordinate tensor (rebuilt each call — atom counts vary)."""
        if self.pdb_path is None:
            return None
        from ._pdb import build_pdb_coords
        features = self._prepare_complex_features(binder, target)
        return build_pdb_coords(
            self.pdb_path, features,
            self.chain_a_len or len(binder),
            self.chain_b_len or len(target),
        )

    def score_complex(
        self, binder: str, target: str, num_loops: int = 0
    ) -> dict:
        """Score a binder-target complex. Returns pTM, ipTM, PAE.

        binder = design candidate (chain A, asym=0)
        target = fixed target    (chain B, asym=1)
        ipTM reflects interface quality between binder and target.

        If pdb_path was set, uses PDB coordinates as x_pred (no diffusion).
        """
        self._ensure_model()

        features = self._prepare_complex_features(binder, target)
        pdb_coords = self._get_pdb_coords(binder, target)

        if self.mode == "score_only":
            from .score_only import score_only_forward
            result = score_only_forward(self._model, features, num_loops=num_loops,
                                        pdb_coords=pdb_coords)
            pae = result["pae"][0].cpu().numpy()
            len_a = features["_len_a"]

            # Per-chain metrics: binder→target PAE
            pae_ab = pae[:len_a, len_a:]  # (binder, target)
            interface_pae = float(pae_ab.mean())

            return {
                "ptm": float(result["ptm"][0].item()),
                "iptm": float(result["iptm"][0].item()) if result["iptm"].numel() else 0.0,
                "pae": pae,
                "interface_pae": interface_pae,
            }
        else:
            # Full diffusion mode
            with torch.inference_mode():
                output = self._model.forward(
                    **{k: v for k, v in features.items() if not k.startswith("_")},
                    num_loops=num_loops,
                    num_sampling_steps=1 if self.mode == "fast" else 200,
                    num_diffusion_samples=1,
                )
            len_a = features["_len_a"]
            pae = output["pae"][0].cpu().numpy()
            pae_ab = pae[:len_a, len_a:]
            return {
                "ptm": float(output["ptm"][0].item()),
                "iptm": float(output["iptm"][0].item()),
                "pae": pae,
                "interface_pae": float(pae_ab.mean()),
            }

    # ---- Batch scoring ----

    def score_complex_batch(
        self, binders: list[str], target: str, num_loops: int = 0
    ) -> list[dict]:
        """Score multiple binder candidates in one batched forward pass.

        All binders must have the same length. Much faster than calling
        score_complex() in a loop (~N x ESMC forward → 1 x ESMC forward).
        """
        self._ensure_model()

        if len(binders) == 0:
            return []
        if len(binders) == 1:
            return [self.score_complex(binders[0], target, num_loops=num_loops)]

        # Prepare per-candidate features
        feats_list = [self._prepare_complex_features(b, target) for b in binders]
        lens_a = [f.pop("_len_a") for f in feats_list]
        pdb_coords = self._get_pdb_coords(binders[0], target)

        if self.mode == "score_only":
            from .score_only import score_only_forward_batch
            result = score_only_forward_batch(self._model, feats_list, num_loops=num_loops)
            # Note: batch mode doesn't support per-candidate PDB coords yet
            # Uses same PDB coords for all (reasonable for scaffold-based design)
            pae_all = result["pae"].cpu().numpy()
            ptm_all = result["ptm"].cpu().numpy()
            iptm_all = result["iptm"].cpu().numpy()

            results = []
            for i, len_a in enumerate(lens_a):
                pae_i = pae_all[i]
                pae_ab = pae_i[:len_a, len_a:]
                results.append({
                    "ptm": float(ptm_all[i]),
                    "iptm": float(iptm_all[i]) if iptm_all.ndim else 0.0,
                    "pae": pae_i,
                    "interface_pae": float(pae_ab.mean()),
                })
            return results
        else:
            # Full diffusion: run one at a time (atom features differ)
            return [self.score_complex(b, target, num_loops) for b in binders]

    def _score_with_diffusion(self, sequence: str, num_loops: int = 1,
                              num_sampling_steps: int | None = None) -> dict:
        """Full pipeline with minimal diffusion steps."""
        if num_sampling_steps is None:
            num_sampling_steps = 1 if self.mode == "fast" else 200

        output = self._model.infer_protein(
            sequence, num_loops=num_loops,
            num_sampling_steps=num_sampling_steps, num_diffusion_samples=1,
        )
        return {
            "plddt": output["plddt"][0].cpu().numpy(),
            "ptm": float(output["ptm"][0].item()),
            "iptm": float(output["iptm"][0].item()) if output.get("iptm") is not None else None,
            "pae": output["pae"][0].cpu().numpy() if output.get("pae") is not None else None,
        }
