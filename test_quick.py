"""Quick install smoke test for EasyNano.

This is the *simple test program* referenced in the README. It only parses a
target PDB and detects CDRs, so it requires NO ESMFold2 model weights — just
`biotite` and `abnumber` (both installed automatically with `pip install -e .`).

Run it with:
    python test_quick.py

Expected output ends with a line like:
    [OK] EasyNano imported and target parsed successfully
"""

from easynano.setup import setup_target_design
from easynano.config import TARGETS


def main():
    name = "ty1_rbd"  # any key from easynano.config.TARGETS works
    cfg = TARGETS[name]
    epitope = [int(x) for x in cfg["epitope"].split(",")] if isinstance(cfg["epitope"], str) else list(cfg["epitope"])

    setup = setup_target_design(
        pdb_path=cfg["pdb"],
        target_chain=cfg["chain"],
        epitope_indices=epitope,
        framework=cfg["framework"],
    )

    print("[OK] EasyNano imported and target parsed successfully")
    print(f"  target   : {cfg['label']}")
    print(f"  target_sequence length : {len(setup['target_sequence'])}")
    print(f"  binder_template length : {len(setup['binder_template'])}")
    print(f"  CDR positions          : {len(setup['cdr_indices'])}")
    print(f"  epitope residues       : {len(setup['epitope_token_indices'])}")
    print(f"  framework              : {setup['framework']}")


if __name__ == "__main__":
    main()
