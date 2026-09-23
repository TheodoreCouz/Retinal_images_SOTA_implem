"""Fill the 3-method x 2-dataset comparison table with mean +/- 95% CI over
seeds 42-46, from the per-run test probabilities in $PROBS_DIR.

One protocol for all three methods (C-Tran/utils/retexpert_metrics.py):
sigmoid probs, 0.5 threshold, all classes, TEST split. macro F1 = f1_score
average='macro'; macro mAP = per-class average_precision macro-averaged;
Recall@n = per-sample recall at n = that sample's own number of true labels
(rank-based, samples with >=1 positive).
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C-Tran'))
from utils.retexpert_metrics import compute_metrics



def recall_at_n(probs, targets):
    """Per-sample recall@n, n = that sample's number of positive labels.
    Same ranking rule as retexpert_metrics.recall_at_k, but k varies per sample
    instead of being fixed at 5. Samples with no positive label are excluded."""
    order = np.argsort(-probs, axis=1)
    scores = []
    for i in range(len(probs)):
        n = int(targets[i].sum())
        if n:
            scores.append(targets[i][order[i, :n]].sum() / n)
    return float(np.mean(scores)) if scores else 0.0


PROBS = os.environ['PROBS_DIR']
SEEDS = (42, 43, 44, 45, 46)
METHODS = [('RETExpert', 'retexpert'), ('RIADD (KAMATALAB)', 'riadd'), ('C-TRAN', 'ctran')]
COLS = [('macro F1', 'f1'), ('macro mAP', 'aupr'), ('Recall@n', 'recall@n')]


def load_riadd(ds, s):
    """KAMATALAB's actual multi-path reconstruction, not the single-model
    stand-in `riadd_{ds}_seed{s}.npz` (dump_test_probs.py) this table used to
    read -- that file predates the 3-path rebuild and isn't the paper's
    method at all (one 5-fold B6 ensemble on the full 29/20-class label set,
    no path 1 / path 2 / path 3 split).

    RFMiD: mean of path1+path2+path3 (riadd3_blend_*, TTA=5/3/3 per path,
    matching the paper's Appendix A.1). MuReD: mean of path2+path3 only --
    path 1's sub-task heads are hardcoded to RFMiD's 29-column layout;
    KAMATALAB never entered MuReD, so there's no path-1 split to reuse for
    it, and dump_riadd23_mured.py accordingly only dumped p2/p3 there."""
    if ds == 'rfmid':
        z = np.load(f'{PROBS}/riadd3_blend_test_seed{s}.npz')
        return z['probs'], z['targets']
    p2 = np.load(f'{PROBS}/riadd23_mured_p2_test_seed{s}.npz')
    p3 = np.load(f'{PROBS}/riadd23_mured_p3_test_seed{s}.npz')
    return (p2['probs'] + p3['probs']) / 2, p2['targets']


def per_seed(method, ds):
    rows = []
    for s in SEEDS:
        if method == 'riadd':
            p, t = load_riadd(ds, s)
        else:
            z = np.load(f'{PROBS}/{method}_{ds}_seed{s}.npz')
            p, t = z['test_pred'], z['test_true']
        m = compute_metrics(p, t, average='macro')
        m['recall@n'] = recall_at_n(p, t)
        rows.append(m)
    return rows


def ci95(vals):
    v = np.asarray(vals, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


results = {(m, d): per_seed(m, d) for _, m in METHODS for d in ('mured', 'rfmid')}

hdr = "| Method | " + " | ".join(f"{c} {d}" for c, _ in COLS for d in ('MuReD', 'RFMiD')) + " |"
print(hdr)
print("|" + "---|" * (1 + 2 * len(COLS)))
for label, m in METHODS:
    cells = []
    for _, key in COLS:
        for d in ('mured', 'rfmid'):
            mean, h = ci95([r[key] for r in results[(m, d)]])
            cells.append(f"{mean:.3f} ± {h:.3f}")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\nper-seed values")
for label, m in METHODS:
    for d in ('mured', 'rfmid'):
        for key, name in [('f1', 'macroF1'), ('aupr', 'macromAP'), ('recall@n', 'Recall@n')]:
            v = [r[key] for r in results[(m, d)]]
            print(f"  {label:18s} {d:6s} {name:9s} " + " ".join(f"{x:.4f}" for x in v))
