"""Dump test-set probabilities for the fixed-loss MuReD retrain (5 seeds).

Same inference as dump_test_probs.py, pointed at the mured_seed<N>_fixedloss/
checkpoints (FDCM-loss and SOA-block-sampling bugs fixed -- see
losses/retexpert_loss.py and utils/engine.py). Writes to a separate tag
(retexpert_mured_fixedloss_seed<N>.npz) so the original checkpoints' dumped
probs are untouched for comparison. MuReD only -- RFMiD was not retrained
(its gamma_FDCM is 0, so the FDCM bug never applied there; the SOA bug's
RFMiD impact was never separately quantified, out of scope for this retrain).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vit_adapter import build_retexpert_vit_large
from utils.datasets import build_dataset

RUNS, DATA, OUT = os.environ['RUNS_DIR'], os.environ['DATA_DIR'], os.environ['PROBS_DIR']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, flush=True)

args = SimpleNamespace(dataset='MuReD', data_path=f'{DATA}/MuReD', input_size=224,
                       color_jitter=0.4, aa='rand-m9-mstd0.5-inc1',
                       reprob=0.25, remode='pixel', recount=1)
loader = torch.utils.data.DataLoader(build_dataset('test', args), batch_size=32,
                                     shuffle=False, num_workers=2)
for seed in (42, 43, 44, 45, 46):
    out = f'{OUT}/retexpert_mured_fixedloss_seed{seed}.npz'
    if os.path.exists(out):
        continue
    model = build_retexpert_vit_large(num_classes=20, adapter_mode='AKU', adapter_dim=64)
    ckpt = torch.load(f'{RUNS}/retexpert/mured_seed{seed}_fixedloss/checkpoint-best.pth',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE).eval()
    probs, targs = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            probs.append(torch.sigmoid(model(x.to(DEVICE))[0]).cpu().numpy())
            targs.append(y.numpy())
    np.savez(out, test_pred=np.concatenate(probs), test_true=np.concatenate(targs))
    print('wrote', out, flush=True)
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
print('DUMP DONE')
