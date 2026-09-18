"""Dump VAL and TEST probabilities of KAMATALAB paths 2 and 3 on MuReD.

Mirrors eval_ci.ensemble_probs (mean sigmoid over the 5 folds) but with the
model and input size parameterised per path, and both splits, so the blend can
be scored under the validation-tuned protocol used for every other row.

  riadd23_mured_p2_{val,test}_seed<N>.npz   B5 @ 960
  riadd23_mured_p3_{val,test}_seed<N>.npz   B6 @ 768
keys: probs (N, 20) float32, targets (N, 20) int64 -- same layout as riadd3_*.
"""
import os
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ci

RUNS, SRC, OUT = os.environ['RUNS_DIR'], os.environ['SRC_DIR'], os.environ['PROBS_DIR']
IMG_DIR = f'{SRC}/MURED/images/images'
SPLITS = {'val': f'{SRC}/MURED/val_labels_stratified.csv',
          'test': f'{SRC}/MURED/test_labels_stratified.csv'}
PATHS = {'p2': ('tf_efficientnet_b5_ns', 960), 'p3': ('tf_efficientnet_b6_ns', 768)}


def load_model(ckpt, arch, nc=20):
    m = create_model(arch, pretrained=False, num_classes=nc)
    ck = torch.load(ckpt, map_location='cpu')
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()}, strict=True)
    return m.cuda().eval()


for path, (arch, img) in PATHS.items():
    for seed in (42, 43, 44, 45, 46):
        ckpts = eval_ci.fold_ckpts(f'{RUNS}/riadd_paths_mured/{path}_seed', seed)
        if ckpts is None:
            print(f'skip {path} seed {seed}: incomplete ensemble', flush=True)
            continue
        for split, csv in SPLITS.items():
            out = f'{OUT}/riadd23_mured_{path}_{split}_seed{seed}.npz'
            if os.path.exists(out):
                continue
            df = pd.read_csv(csv)
            y = df[df.columns[1:]].values.astype(np.int64)
            ds = PrismDataSet(image_ids=df, baseImgPath=IMG_DIR)
            ds.transform = get_riadd_valid_transforms(SimpleNamespace(img_size=img))
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
            fold_p = []
            for cp in ckpts:
                m = load_model(cp, arch)
                fold_p.append(eval_ci.predict(m, loader))
                del m; torch.cuda.empty_cache()
            np.savez(out, probs=np.mean(fold_p, axis=0).astype(np.float32), targets=y)
            print('wrote', out, np.mean(fold_p, axis=0).shape, flush=True)
print('DUMP DONE')
