"""RFMiD: how much of the KAMATALAB F1 gap each upstream inference step closes.

Two factors, ablated independently:
  TTA        -- 3 stochastic passes per fold through get_riadd_test_transforms,
                as upstream subnmit_riadd.py does (CFG['tta']=3).
  thresholds -- per-class F1-optimal cut-points tuned on the RFMiD VALIDATION
                split and applied to test, via metrics9.tune_thresholds. Upstream
                emits raw probabilities and never picks a threshold, so the 0.5
                cutoff was our choice; it is what leaves the rare classes silent.

Test metrics only; validation is used solely to pick thresholds.
"""
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Pytorch-RIADD'))
from metrics9 import tune_thresholds

PROBS = os.environ['PROBS_DIR']
SEEDS = (42, 43, 44, 45, 46)
KAMATA = {'DR AUC': .9875, 'SC1 F1': .9625, 'MD Avg': .7821, 'SC2 F1': .5546, 'Final': .8848}


def macro(fn, p, t):
    return float(np.mean([fn(t[:, c], p[:, c]) for c in range(t.shape[1])
                          if len(np.unique(t[:, c])) > 1]))


def macro_f1(p, t, thr):
    return float(np.mean([f1_score(t[:, c].astype(int), (p[:, c] >= thr[c]).astype(int),
                                   zero_division=0) for c in range(t.shape[1])]))


def n_silent(p, t, thr):
    return sum(1 for c in range(t.shape[1]) if (p[:, c] >= thr[c]).sum() == 0)


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


rows = {}
for tta in ('notta', 'tta'):
    for mode in ('0.5', 'tuned'):
        acc = {k: [] for k in KAMATA}
        acc['silent'] = []
        for s in SEEDS:
            va = np.load(f'{PROBS}/riadd_rfmid_val_{tta}_seed{s}.npz')
            te = np.load(f'{PROBS}/riadd_rfmid_test_{tta}_seed{s}.npz')
            vp, vy, p, t = va['probs'], va['targets'], te['probs'], te['targets']
            thr = np.full(29, 0.5) if mode == '0.5' else tune_thresholds(vp, vy)

            acc['DR AUC'].append(roc_auc_score(t[:, 0], p[:, 0]))
            acc['SC1 F1'].append(f1_score(t[:, 0].astype(int),
                                          (p[:, 0] >= thr[0]).astype(int), zero_division=0))
            acc['SC2 F1'].append(macro_f1(p[:, 1:], t[:, 1:], thr[1:]))
            md = 0.5 * (macro(roc_auc_score, p[:, 1:], t[:, 1:])
                        + macro(average_precision_score, p[:, 1:], t[:, 1:]))
            acc['MD Avg'].append(md)
            acc['Final'].append(0.5 * (acc['DR AUC'][-1] + md))
            acc['silent'].append(n_silent(p[:, 1:], t[:, 1:], thr[1:]))
        rows[(tta, mode)] = acc

hdr = f"{'inference':22s}" + "".join(f"{k:>17s}" for k in KAMATA) + f"{'silent/28':>11s}"
print(hdr)
print('-' * len(hdr))
for (tta, mode), acc in rows.items():
    label = f"{'TTA x3' if tta == 'tta' else 'no TTA':7s} + thr {mode}"
    cells = ""
    for k in KAMATA:
        m, h = ci(acc[k])
        cells += f"  {m*100:6.2f} ±{h*100:5.2f}"
    print(f"{label:22s}{cells}{np.mean(acc['silent']):9.1f}")
print(f"{'KAMATALAB (paper)':22s}" + "".join(f"  {v*100:6.2f}       " for v in KAMATA.values()))
