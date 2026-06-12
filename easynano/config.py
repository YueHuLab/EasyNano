"""Central configuration for EasyNano.

All paths can be overridden via environment variables.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model paths (set env vars or edit defaults)
# ---------------------------------------------------------------------------
ESMFOLD2_FAST = os.environ.get("ESMFOLD2_FAST", "/Users/huyue/esm-c-fold2/ESMFold2-Fast")
ESMFOLD2_FULL = os.environ.get("ESMFOLD2_FULL", "/Users/huyue/esm-c-fold2/ESMFold2")
ESMC_PATH     = os.environ.get("ESMC_PATH", "/Users/huyue/esm-c-fold2/ESMC-6B")
DEVICE        = os.environ.get("EASYNANO_DEVICE", "mps")
ESM_REPO      = os.environ.get("ESM_REPO", "/Users/huyue/esm-c-fold2")

# ---------------------------------------------------------------------------
# Design defaults (manuscript sweet spot — see PARAMETER_SWEEP_FINDINGS.md)
# ---------------------------------------------------------------------------
N_STEPS         = 60
LR              = 0.05
WT_LOGIT        = 2.0    # WT AA logit bias at init (critical param)
W_EPITOPE       = 0.2
W_INTRA         = 0.5
W_INTER         = 0.5
W_GLOB          = 0.2
W_PRIOR         = 0.05   # sweet spot: prior=0.05
W_AA_FREQ       = 0.01
TEMP_MIN        = 0.1
LOG_EVERY       = 4
SNAPSHOT_EVERY  = 4
SAMPLE_STEPS    = 5      # Fast model forward sampling steps
N_LOOPS         = 1      # Fast model recycling loops

# Full model evaluation defaults
FULL_LOOPS      = 3
FULL_SAMPLING   = 14
FULL_DIFFUSION  = 4      # for structure prior

# ---------------------------------------------------------------------------
# Prior defaults
# ---------------------------------------------------------------------------
N_BINS          = 64
MIN_DIST        = 2.0
MAX_DIST        = 22.0
BIN_TOLERANCE   = 2.5
EPITOPE_CUTOFF  = 8.0

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------
TOKENS = ["<pad>", "-", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
          "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_TO_TOKEN = {aa: i for i, aa in enumerate(TOKENS)}
CYS_TOK = AA_TO_TOKEN["C"]
NUM_RES_TYPES = 33
AA_DIMS = 20
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_NO_CYS = "ADEFGHIKLMNPQRSTVWY"

AA_FREQ = [
    0.0743, 0.0510, 0.0443, 0.0477, 0.0290, 0.0399, 0.0604, 0.0677,
    0.0227, 0.0554, 0.0968, 0.0580, 0.0221, 0.0394, 0.0444, 0.0580,
    0.0537, 0.0127, 0.0300, 0.0660,
]

# ---------------------------------------------------------------------------
# Framework init sequences (CDR positions are mutable in design)
# ---------------------------------------------------------------------------
INIT_FRAMEWORKS: dict[str, str] = {
    "b5": "QVQLVESGGGLVQPGGSLRLSCAASGLQIGYGVYMSLGWFRQAPGQGLEAVAAISYSGQSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAARVVTDSYQPIYKAPIRWGQGTLVTVS",
    "vhh72": "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSSGS",
    "kn035": "QVQLQESGGGLVQPGGSLRLSCAASGKMSSRRCMAWFRQAPGKERERVAKLLTTSGSTYLADSVKGRFTISQNNAKSTVYLQMNSLKPEDTAMYYCAADSFEDPTCTLVTSSGAFQYWGQGTQVTVS",
    "antitnf": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSNYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLKPEDTALYYCARSPSGFNRGQGTQVTVSS",
    "ty1": "QVQLVETGGGLVQPGGSLRLSCAASGFTFSSVYMNWVRQAPGKGPEWVSRISPNSGNIGYTDSVKGRFTISRDNAKNTLYLQMNNLKPEDTALYYCAIGLNLSSSSVRGQGTQVTVSS",
    "vhh3": "QLQESGGGLVQPGGSLRLSCAASGRTFSDHSGYTYTIGWFRQAPGKEREFVARIYWSSGNTYYADSVKGRFAISRDIAKNTVDLTMNNLEPEDTAVYYCAARDGIPTSRSVESYNYWGQGTQVTVSS",
}

# ---------------------------------------------------------------------------
# Target registry — all 6 manuscript targets
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict] = {
    "ty1_rbd": {
        "pdb": "test/6ZXN.pdb",
        "chain": "A",
        "epitope": "18,19,22,116,117,118,119,120,122,138,140,152,153,154,155,156,159,160,162,163,164",
        "framework": "ty1",
        "label": "Ty1/RBD (6ZXN)",
        "allow_cdr_cys": False,
    },
    "vhh72_rbd": {
        "pdb": "test/6WAQ.pdb",
        "chain": "B",
        "epitope": "35,36,37,38,39,40,41,42,43,44,45,46,49,50",
        "framework": "vhh72",
        "label": "VHH72/RBD (6WAQ)",
        "allow_cdr_cys": False,
    },
    "kn035_pdl1": {
        "pdb": "test/5JDS.pdb",
        "chain": "A",
        "epitope": "36,38,43,45,48,50,97,98,99,101,102,103,104,105",
        "framework": "kn035",
        "label": "KN035/PD-L1 (5JDS)",
        "allow_cdr_cys": True,
    },
    "antitnf_tnfa": {
        "pdb": "test/5M2J.pdb",
        "chain": "A",
        "epitope": "66,67,68,79,80,81,82,83,84,117,118",
        "framework": "antitnf",
        "label": "anti-TNF/TNFα (5M2J)",
        "allow_cdr_cys": False,
    },
    "vhh3_tnfa": {
        "pdb": "test/5M2M.pdb",
        "chain": "B",
        "epitope": "15,16,17,18,58,59,60,61,62,63,99,100,130,131,132,133,135",
        "framework": "vhh3",
        "label": "VHH3/TNFα (5M2M)",
        "allow_cdr_cys": False,
    },
    "b5_aqp4": {
        "pdb": "test/B5.pdb",
        "chain": "A",
        "epitope": "0,1,2,3,4,5,6,7,8,9,10",
        "framework": "b5",
        "label": "B5/AQP4 (de novo)",
        "allow_cdr_cys": False,
    },
}

# Loss weights dict (used by loss.py, overridden per-run)
LOSS_WEIGHTS: dict[str, float] = {
    "intra_contact": 0.5,
    "inter_contact": 0.5,
    "glob": 0.2,
    "epitope": 0.2,
    "structure_prior": 0.05,
}

MUTABLE_TOKEN: str = "#"
