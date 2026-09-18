"""Dump VALIDATION-set sigmoid probabilities for each finished RetExpert run.

dump_test_probs.py wrote only the test split, which forced every table that
includes RetExpert onto a fixed 0.5 threshold. With validation probabilities the
per-class thresholds can be tuned exactly as for the other models, so one
protocol serves every row. Output is a separate file (…_val.npz) so the
test-split dumps used elsewhere are never rewritten.
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
for name, nc, tag in [('MuReD', 20, 'mured'), ('RFMiD', 29, 'rfmid')]:
    args = SimpleNamespace(dataset=name, data_path=f'{DATA}/{name}', input_size=224,
                           color_jitter=0.4, aa='rand-m9-mstd0.5-inc1',
                           reprob=0.25, remode='pixel', recount=1)
    loader = torch.utils.data.DataLoader(build_dataset('val', args), batch_size=32,
                                         shuffle=False, num_workers=8)
    for seed in (42, 43, 44, 45, 46):
        out = f'{OUT}/retexpert_{tag}_seed{seed}_val.npz'
        if os.path.exists(out):
            continue
        model = build_retexpert_vit_large(num_classes=nc, adapter_mode='AKU', adapter_dim=64)
        ckpt = torch.load(f'{RUNS}/retexpert/{tag}_seed{seed}/checkpoint-best.pth',
                          map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.cuda().eval()
        probs, targs = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                probs.append(torch.sigmoid(model(x.cuda())[0]).cpu().numpy())
                targs.append(y.numpy())
        np.savez(out, val_pred=np.concatenate(probs), val_true=np.concatenate(targs))
        print('wrote', out, np.concatenate(probs).shape, flush=True)
        del model
        torch.cuda.empty_cache()
print('DUMP DONE')
