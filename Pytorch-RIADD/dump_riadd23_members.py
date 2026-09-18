"""Per-MEMBER test probabilities of the Table-1 RIADD system (paths 2 + 3).

The existing dumps average the 5 folds of each path; the between-member
correlation needs every member on its own. One deterministic pass per model
(get_riadd_valid_transforms) — the stochastic TTA would only add noise to a
correlation estimate.

  riadd23_members_{mured,rfmid}_test_seed<N>.npz
      probs   (10, N, C) float32  members ordered p2 fold0..4, p3 fold0..4
      targets (N, C) int64
      members list of "p2/fold0" ... strings
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet, RiaddDataSet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ci

RUNS, SRC, OUT = os.environ['RUNS_DIR'], os.environ['SRC_DIR'], os.environ['PROBS_DIR']
PATHS = {'p2': ('tf_efficientnet_b5_ns', 960), 'p3': ('tf_efficientnet_b6_ns', 768)}
DATASETS = {
    'mured': dict(nc=20, ds=PrismDataSet, img_dir=f'{SRC}/MURED/images/images',
                  csv=f'{SRC}/MURED/test_labels_stratified.csv',
                  prefix=f'{RUNS}/riadd_paths_mured/{{p}}_seed'),
    'rfmid': dict(nc=29, ds=RiaddDataSet, img_dir=f'{SRC}/RFMiD/Test',
                  csv=f'{SRC}/RFMiD/testing_labels_29.csv',
                  prefix=f'{RUNS}/riadd_paths/{{p}}_full_seed'),
}


def load_model(ckpt, arch, nc):
    m = create_model(arch, pretrained=False, num_classes=nc)
    ck = torch.load(ckpt, map_location='cpu')
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()}, strict=True)
    return m.cuda().eval()


for name, cfg in DATASETS.items():
    df = pd.read_csv(cfg['csv'])
    y = df[df.columns[1:]].values.astype(np.int64)
    for seed in (42, 43, 44, 45, 46):
        out = f'{OUT}/riadd23_members_{name}_test_seed{seed}.npz'
        if os.path.exists(out):
            continue
        probs, members = [], []
        for path, (arch, img) in PATHS.items():
            ckpts = eval_ci.fold_ckpts(cfg['prefix'].format(p=path), seed)
            if ckpts is None:
                print(f'skip {name} {path} seed {seed}: incomplete', flush=True)
                break
            ds = cfg['ds'](image_ids=df, baseImgPath=cfg['img_dir'])
            ds.transform = get_riadd_valid_transforms(SimpleNamespace(img_size=img))
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
            for k, cp in enumerate(ckpts):
                m = load_model(cp, arch, cfg['nc'])
                probs.append(eval_ci.predict(m, loader).astype(np.float32))
                members.append(f'{path}/fold{k}')
                del m; torch.cuda.empty_cache()
        if len(probs) == 10:
            np.savez(out, probs=np.stack(probs), targets=y, members=np.array(members))
            print('wrote', out, flush=True)
print('DUMP DONE')
