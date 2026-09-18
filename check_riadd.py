"""Check the rebuilt Pytorch-RIADD runs against (a) the previous 2-GPU results and
(b) the RIADD challenge paper.

(a) uses the same protocol as make_table.py -- macro F1 at 0.5, macro mAP, Recall@n.
(b) uses the challenge's own score, documented in RFMiD/training_classes.txt:
      Multi-Disease Avg = 0.5 * (mean AUC + mAP) over the disease columns
      Final Score       = 0.5 * (Disease_Risk AUC + Multi-Disease Avg)
"""
import os

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROBS = os.environ['PROBS_DIR']
SEEDS = (42, 43, 44, 45, 46)

# MuReD, same protocol as make_table.py. "2-GPU" = the previous validated runs,
# "1-GPU b2" = the superseded runs with the halved BatchNorm batch.
REF = {'macro F1': (.6287, .524), 'macro mAP': (.7110, .592), 'Recall@n': (.7823, .715)}
# KAMATALAB, RIADD challenge test set (paper Table 7).
KAMATA = {'Disease_Risk AUC': .9875, 'Multi-Disease Avg': .7821, 'Final Score': .8848}


def macro(fn, p, t):
    v = [fn(t[:, c], p[:, c]) for c in range(t.shape[1]) if len(np.unique(t[:, c])) > 1]
    return float(np.mean(v))


def recall_at_n(p, t):
    o = np.argsort(-p, axis=1)
    s = [t[i][o[i, :int(t[i].sum())]].sum() / t[i].sum() for i in range(len(p)) if t[i].sum()]
    return float(np.mean(s))


def ci95(v):
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


print("=== MuReD, 5 seeds: rebuilt vs previous ===")
vals = {k: [] for k in REF}
for s in SEEDS:
    z = np.load(f'{PROBS}/riadd_mured_seed{s}.npz')
    p, t = z['test_pred'], z['test_true']
    vals['macro F1'].append(f1_score(t.astype(int), (p >= .5).astype(int),
                                     average='macro', zero_division=0))
    vals['macro mAP'].append(macro(average_precision_score, p, t))
    vals['Recall@n'].append(recall_at_n(p, t))
print(f"{'metric':11s} {'rebuilt (95% CI)':26s} {'prev 2-GPU':>11s} {'broken 1-GPU':>13s}")
for k, v in vals.items():
    m, h = ci95(v)
    old, bad = REF[k]
    print(f"{k:11s} {m*100:6.2f}%  [{(m-h)*100:5.2f}, {(m+h)*100:5.2f}]   "
          f"{old*100:9.2f}%  {bad*100:11.2f}%   (vs prev {(m-old)*100:+.2f})")
print("per-seed macro F1: " + " ".join(f"{x*100:.2f}" for x in vals['macro F1']))

print("\n=== RFMiD challenge score vs KAMATALAB (paper Table 7, test set) ===")
done = [s for s in SEEDS if os.path.exists(f'{PROBS}/riadd_rfmid_seed{s}.npz')]
for s in done:
    z = np.load(f'{PROBS}/riadd_rfmid_seed{s}.npz')
    p, t = z['test_pred'], z['test_true']
    dr = roc_auc_score(t[:, 0], p[:, 0])
    md = 0.5 * (macro(roc_auc_score, p[:, 1:], t[:, 1:])
                + macro(average_precision_score, p[:, 1:], t[:, 1:]))
    got = {'Disease_Risk AUC': dr, 'Multi-Disease Avg': md, 'Final Score': 0.5 * (dr + md)}
    print(f"  seed {s} (of {len(SEEDS)}): " + "  ".join(
        f"{k} {v*100:.2f}% (paper {KAMATA[k]*100:.2f}%, {(v-KAMATA[k])*100:+.2f})"
        for k, v in got.items()))
if len(done) < len(SEEDS):
    print(f"  [{len(SEEDS)-len(done)} RFMiD seeds still training -- no CI yet]")
