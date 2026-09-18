"""Three-path KAMATALAB reproduction vs the challenge paper (Table 7, test set).

Reports each path on its own and the equal-weight blend, on the challenge's own
metrics. Threshold 0.5 is the REPRODUCTION number -- upstream emits raw
probabilities and never picks a threshold, so 0.5 is the neutral default and the
one the other methods in the comparison table use. The validation-tuned row is
an ABLATION showing how much of the residual is threshold choice, not a
reproduction result.
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
VARIANTS = [('p1', 'path 1  B5@768 multi-stage'), ('p2', 'path 2  B5@960 direct'),
            ('p3', 'path 3  B6@768 direct'), ('blend', 'BLEND of all three')]


def macro(fn, p, t):
    return float(np.mean([fn(t[:, c], p[:, c]) for c in range(t.shape[1])
                          if len(np.unique(t[:, c])) > 1]))


def scores(p, t, thr):
    dr = roc_auc_score(t[:, 0], p[:, 0])
    md = 0.5 * (macro(roc_auc_score, p[:, 1:], t[:, 1:])
                + macro(average_precision_score, p[:, 1:], t[:, 1:]))
    sc2 = float(np.mean([f1_score(t[:, c].astype(int), (p[:, c] >= thr[c]).astype(int),
                                  zero_division=0) for c in range(1, 29)]))
    return {'DR AUC': dr, 'SC1 F1': f1_score(t[:, 0].astype(int),
                                             (p[:, 0] >= thr[0]).astype(int), zero_division=0),
            'MD Avg': md, 'SC2 F1': sc2, 'Final': 0.5 * (dr + md)}


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


for mode in ('0.5', 'tuned'):
    print(f"\n=== threshold {mode}"
          f"{'   (REPRODUCTION)' if mode == '0.5' else '   (ablation, not upstream)'} ===")
    print(f"{'variant':28s}" + "".join(f"{k:>16s}" for k in KAMATA) + f"{'silent/28':>11s}")
    for key, label in VARIANTS:
        acc, sil = {k: [] for k in KAMATA}, []
        for s in SEEDS:
            f = f'{PROBS}/riadd3_{key}_test_seed{s}.npz'
            if not os.path.exists(f):
                continue
            te = np.load(f)
            p, t = te['probs'], te['targets']
            if mode == '0.5':
                thr = np.full(29, 0.5)
            else:
                va = np.load(f'{PROBS}/riadd3_{key}_val_seed{s}.npz')
                thr = tune_thresholds(va['probs'], va['targets'])
            for k, v in scores(p, t, thr).items():
                acc[k].append(v)
            sil.append(sum(1 for c in range(1, 29) if (p[:, c] >= thr[c]).sum() == 0))
        if not sil:
            print(f"{label:28s}   (not available yet)")
            continue
        cells = "".join(f"  {ci(acc[k])[0]*100:6.2f} ±{ci(acc[k])[1]*100:4.2f}" for k in KAMATA)
        print(f"{label:28s}{cells}{np.mean(sil):9.1f}   n={len(sil)}")
    print(f"{'KAMATALAB (paper Table 7)':28s}" + "".join(f"  {v*100:6.2f}      " for v in KAMATA.values()))
