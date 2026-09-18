"""Dump 5-fold-ensemble test probabilities for each finished Pytorch-RIADD run.

Reuses eval_ci's ensemble_probs (mean sigmoid over the 5 folds, cached per
seed) with this deployment's paths, and the stratified MURED test split the
other two methods are scored on.
"""
import os

import numpy as np

import eval_ci

RUNS, SRC, OUT = os.environ['RUNS_DIR'], os.environ['SRC_DIR'], os.environ['PROBS_DIR']
eval_ci.PRED_DIR = f'{OUT}/riadd_folds'
os.makedirs(eval_ci.PRED_DIR, exist_ok=True)

CFG = {
    'mured': dict(prefix=f'{RUNS}/riadd/mured_seed', nc=20, ds=eval_ci.PrismDataSet,
                  test=(f'{SRC}/MURED/images/images',
                        f'{SRC}/MURED/test_labels_stratified.csv')),
    'rfmid': dict(prefix=f'{RUNS}/riadd/rfmid_seed', nc=29, ds=eval_ci.RiaddDataSet,
                  test=(f'{SRC}/RFMiD/Test', f'{SRC}/RFMiD/testing_labels_29.csv')),
}

for tag, cfg in CFG.items():
    for seed in (42, 43, 44, 45, 46):
        out = f'{OUT}/riadd_{tag}_seed{seed}.npz'
        if os.path.exists(out):
            continue
        probs, targs, _ = eval_ci.ensemble_probs(cfg, seed, 'test')
        if probs is None:                      # some of the 5 folds not trained yet
            print(f'skip {tag} seed {seed}: incomplete ensemble', flush=True)
            continue
        np.savez(out, test_pred=probs, test_true=targs)
        print('wrote', out, flush=True)
