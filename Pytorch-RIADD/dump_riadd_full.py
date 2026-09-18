"""Dump RFMiD ensemble probabilities the way the upstream repo infers.

Upstream subnmit_riadd.py averages, per fold model, CFG['tta']=3 stochastic
passes through get_riadd_test_transforms (random h-flip / HSV / brightness at
p=0.5), then blends the folds with equal weights. Our evaluation so far used a
single deterministic get_riadd_valid_transforms pass, so this adds the missing
TTA and writes both variants side by side.

Also dumps the VALIDATION split, which the test-time thresholds are tuned on --
the upstream submission emits raw probabilities and never picks a threshold, so
0.5 was our choice, not theirs, and it is what starves the rare classes.

Writes <PROBS_DIR>/riadd_rfmid_{split}_{tta|notta}_seed{S}.npz
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import eval_ci
from timm.data import get_riadd_test_transforms, get_riadd_valid_transforms

RUNS, SRC, OUT = os.environ['RUNS_DIR'], os.environ['SRC_DIR'], os.environ['PROBS_DIR']
SEEDS = (42, 43, 44, 45, 46)
IMG = 960
TTA = 3                      # CFG['tta'] upstream
BS = 8

SPLITS = {
    'val':  (f'{SRC}/RFMiD/Validation', f'{SRC}/RFMiD/validation_labels_29.csv'),
    'test': (f'{SRC}/RFMiD/Test',       f'{SRC}/RFMiD/testing_labels_29.csv'),
}


def ensemble(seed, split, use_tta):
    img_dir, csv = SPLITS[split]
    df = pd.read_csv(csv)
    y = df[df.columns[1:]].values.astype(np.int64)
    ckpts = eval_ci.fold_ckpts(f'{RUNS}/riadd/rfmid_seed', seed)
    if ckpts is None:
        return None, y
    ds = eval_ci.RiaddDataSet(image_ids=df, baseImgPath=img_dir)
    # get_riadd_test_transforms indexes args like a dict; the valid one uses attrs.
    ds.transform = (get_riadd_test_transforms({'img_size': IMG}) if use_tta
                    else get_riadd_valid_transforms(SimpleNamespace(img_size=IMG)))
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=6, pin_memory=True)

    passes = []
    for cp in ckpts:
        m = eval_ci.load_model(cp, 29)
        for _ in range(TTA if use_tta else 1):
            passes.append(eval_ci.predict(m, loader))
        del m
        torch.cuda.empty_cache()
    return np.mean(passes, axis=0), y


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    # $RIADD_VARIANTS selects which of the two to compute, so the cheap non-TTA
    # pass (which is all the threshold tuning needs) can run on the short debug
    # partition without waiting behind the 3x-costlier TTA pass.
    want = os.environ.get('RIADD_VARIANTS', 'notta,tta').split(',')
    for use_tta in (False, True):
        tag = 'tta' if use_tta else 'notta'
        if tag not in want:
            continue
        for split in SPLITS:
            for seed in SEEDS:
                out = f'{OUT}/riadd_rfmid_{split}_{tag}_seed{seed}.npz'
                if os.path.exists(out):
                    continue
                p, y = ensemble(seed, split, use_tta)
                if p is None:
                    print(f'skip {split} {tag} seed {seed}: incomplete', flush=True)
                    continue
                np.savez(out, probs=p, targets=y)
                print('wrote', out, flush=True)
