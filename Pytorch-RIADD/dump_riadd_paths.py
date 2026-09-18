"""Assemble KAMATALAB's three-path system and dump its blended probabilities.

Per path, per seed: average the 5 fold models, each run TTA x3 through
get_riadd_test_transforms, exactly as upstream subnmit_riadd.py does
(CFG['tta']=3, equal fold weights).

Path one is multi-stage, so its 29 columns are stitched from four separately
trained heads. The column indices are upstream's own (the iloc lists inside
RiaddDataSet{9,8,11}Classes), shifted by one because our probability matrix
drops the ID column: CSV column k -> probability index k-1.

  index 0        Disease_Risk   <- 'dr' head, output column 0
  indices 1..28  the 28 diseases <- g9 / g8 / g11 heads

Writes, per seed and split: one npz per path plus the equal-weight blend.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import eval_ci
from timm.models import create_model
from timm.data import get_riadd_test_transforms, RiaddDataSet

RUNS, SRC, OUT = os.environ['RUNS_DIR'], os.environ['SRC_DIR'], os.environ['PROBS_DIR']
SEEDS = (42, 43, 44, 45, 46)
TTA = 3
BS = 8

SPLITS = {
    'val':  (f'{SRC}/RFMiD/Validation', f'{SRC}/RFMiD/validation_labels_29.csv'),
    'test': (f'{SRC}/RFMiD/Test',       f'{SRC}/RFMiD/testing_labels_29.csv'),
}

# CSV iloc indices from the vendored RiaddDataSet*Classes, minus 1.
G9  = [i - 1 for i in (2, 3, 4, 5, 6, 7, 8, 13, 17)]
G8  = [i - 1 for i in (10, 12, 14, 18, 23, 24, 26, 29)]
G11 = [i - 1 for i in (9, 11, 15, 16, 19, 20, 21, 22, 25, 27, 28)]

PATHS = {
    # path: (model, img_size, [(subtask, n_out, target_columns), ...])
    'p1': ('tf_efficientnet_b5_ns', 768, [('dr', 2, [0]), ('g9', 9, G9),
                                          ('g8', 8, G8), ('g11', 11, G11)]),
    'p2': ('tf_efficientnet_b5_ns', 960, [('full', 29, list(range(29)))]),
    'p3': ('tf_efficientnet_b6_ns', 768, [('full', 29, list(range(29)))]),
}


def load(ckpt, model_name, nc):
    m = create_model(model_name, pretrained=False, num_classes=nc)
    ck = torch.load(ckpt, map_location='cpu')
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()},
                      strict=True)
    return m.cuda().eval()


def head_probs(path, subtask, nc, seed, loader):
    """Mean sigmoid over 5 folds x TTA passes for one head."""
    model_name, _, _ = PATHS[path]
    ckpts = eval_ci.fold_ckpts(f'{RUNS}/riadd_paths/{path}_{subtask}_seed', seed)
    if ckpts is None:
        return None
    out = []
    for cp in ckpts:
        m = load(cp, model_name, nc)
        for _ in range(TTA):
            out.append(eval_ci.predict(m, loader))
        del m
        torch.cuda.empty_cache()
    return np.mean(out, axis=0)


def path_probs(path, seed, split):
    _, img, heads = PATHS[path]
    img_dir, csv = SPLITS[split]
    df = pd.read_csv(csv)
    ds = RiaddDataSet(image_ids=df, baseImgPath=img_dir)
    ds.transform = get_riadd_test_transforms({'img_size': img})
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=6, pin_memory=True)

    full = np.zeros((len(df), 29), dtype=np.float64)
    for subtask, nc, cols in heads:
        p = head_probs(path, subtask, nc, seed, loader)
        if p is None:
            return None, None
        full[:, cols] = p[:, :len(cols)]     # 'dr' keeps only its column 0
    return full, df[df.columns[1:]].values.astype(np.int64)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    for split in SPLITS:
        for seed in SEEDS:
            per_path = {}
            for path in PATHS:
                out = f'{OUT}/riadd3_{path}_{split}_seed{seed}.npz'
                if os.path.exists(out):
                    per_path[path] = np.load(out)['probs']
                    continue
                p, y = path_probs(path, seed, split)
                if p is None:
                    print(f'skip {path} {split} seed {seed}: incomplete', flush=True)
                    continue
                np.savez(out, probs=p, targets=y)
                per_path[path] = p
                print('wrote', out, flush=True)
            if len(per_path) == len(PATHS):
                _, y = SPLITS[split][1], None
                y = pd.read_csv(SPLITS[split][1])
                y = y[y.columns[1:]].values.astype(np.int64)
                blend = np.mean([per_path[p] for p in PATHS], axis=0)
                bo = f'{OUT}/riadd3_blend_{split}_seed{seed}.npz'
                np.savez(bo, probs=blend, targets=y)
                print('wrote', bo, flush=True)
    print('PATHS DUMP DONE')
