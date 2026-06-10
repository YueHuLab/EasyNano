"""Re-fold the top designs with high-quality sampling and compare."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/Users/huyue/esm-c-fold2')
import torch
import torch.nn.functional as F
from binder_design_hy_losses import MUTABLE_TOKEN, compute_structure_losses, get_mid_points
from test_b5_pdb import setup_design

from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
config = ESMFold2Config.from_pretrained('/Users/huyue/esm-c-fold2/ESMFold2-Fast')
config.esmc_id = '/Users/huyue/esm-c-fold2/ESMC-6B'
model = ESMFold2Model.from_pretrained('/Users/huyue/esm-c-fold2/ESMFold2-Fast', config=config).float().to('mps').eval()
from esmscore.score_only import _patch_for_mps
_patch_for_mps(model)
unwrapped = model.forward
while hasattr(unwrapped, '__wrapped__'):
    unwrapped = unwrapped.__wrapped__
model.forward = unwrapped.__get__(model, type(model))
print('Model loaded', flush=True)

setup = setup_design(epitope_cutoff=8.0, prior_min_dist=2.5)
target_seq = setup['target_sequence']
target_len = len(target_seq)
binder_template = setup['binder_template']
binder_len = len(binder_template)
epi = setup['epitope_token_indices']
cdr = setup['cdr_indices']
prior_bins = setup['prior_bins']
prior_mask = setup['prior_mask']

sub_designs = {
    'WT (initial)':         'GFTFGTGSYYSSSGSSRGFTYSYYPDYRAYDF',
    'Best by epi (s23)':    'GFKAGTGTYTSSSQSNRGANYGFNPDYRVYLK',
    'Best by total (s27)':  'GFKVGTGTYWSASQTNRGANYGLNPDYRVYLK',
    'Final (s30)':          'GFKAGTGNYWSASQTNRGANYGLNPDYRVYLK',
}

def inject(sub_cdr, template, cdr_indices):
    chars = list(template)
    for i, c in zip(cdr_indices, sub_cdr):
        chars[i] = c
    return ''.join(chars)

def fold_hard(binder_seq, target_seq, num_sampling_steps=5, num_loops=2):
    from esmscore._complex import build_complex_features
    feats = build_complex_features(binder_seq, target_seq)
    features = {k: v for k, v in feats.items() if not k.startswith('_')}
    features = {k: (v.to('mps') if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
    with torch.no_grad():
        out = model.forward(**features, num_loops=num_loops,
                            num_sampling_steps=num_sampling_steps, num_diffusion_samples=1)
    return out

def expected_distance(disto):
    midpoints = get_mid_points(n_bins=64, min_dist=2.0, max_dist=22.0).to(disto.device)
    probs = torch.softmax(disto, dim=-1)
    return (probs * midpoints).sum(-1)

print(f'\n{"Name":<24} {"pTM":>6} {"ipTM":>6} {"CDR→epi":>9} {"prior":>6} {"inter":>6} {"intra":>6} {"total":>7}', flush=True)
print('-' * 90, flush=True)
for name, sub_cdr in sub_designs.items():
    full_binder = inject(sub_cdr, binder_template, cdr)
    assert len(full_binder) == binder_len, f'{len(full_binder)} vs {binder_len}'
    out = fold_hard(full_binder, target_seq, num_sampling_steps=5, num_loops=2)
    disto = out['distogram_logits'].float()
    perm = torch.cat([torch.arange(binder_len, disto.size(1)), torch.arange(0, binder_len)])
    disto_tf = disto[:, perm, :, :][:, :, perm, :]
    losses = compute_structure_losses(
        disto_tf, binder_length=binder_len,
        epitope_token_indices=epi, cdr_indices=cdr,
        prior_bins=prior_bins, prior_mask=prior_mask,
        n_bins=64, min_dist=2.0, max_dist=22.0,
    )
    ptm = float(out['ptm'][0].item()) if out['ptm'].numel() else 0.0
    iptm = float(out['iptm'][0].item()) if out['iptm'].numel() else 0.0
    e_dist = expected_distance(disto_tf)[0]
    cross = e_dist[target_len:, :target_len]
    cdr_to_e = cross[cdr][:, epi]
    cdr_min = cdr_to_e.min(dim=-1).values.mean().item()
    print(f'{name:<24} {ptm:>6.3f} {iptm:>6.3f} {cdr_min:>9.2f} '
          f'{losses["structure_prior_loss"].item():>6.3f} '
          f'{losses["inter_contact_loss"].item():>6.3f} '
          f'{losses["intra_contact_loss"].item():>6.3f} '
          f'{losses["total_loss"].item():>7.3f}', flush=True)
print(flush=True)
