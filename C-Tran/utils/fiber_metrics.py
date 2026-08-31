"""
Fiber evaluation protocol — the 9 metrics defined in metrics.txt.

Family A (macro, threshold-based, per-class F1-optimal threshold tuned on val):
  1. macro F1   2. macro sensitivity (=macro recall)   3. macro specificity
Family B (macro, threshold-free):
  4. macro AUROC   5. macro mAP
Family C (per-image, rank-based, threshold-free):
  6. Recall@n (cut-off n_i = image's own positive count = R-precision)
  7-9. TNR@(m-n)-j, j=1,2,3 (bottom (#neg - j) classes)

Mechanics mirror Fiber/fiber/evaluation/test.py (_tune_thresholds_fbeta,
per-image argsort); the metric *set* follows metrics.txt (authoritative).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def tune_thresholds(val_true, val_prob, default=0.5):
    """Per-class F1-optimal threshold on validation (metrics.txt family-A rule).

    Candidates = midpoints between consecutive sorted unique val probs of the
    class + {0,1} sentinels; t_c = argmax per-class F1. Classes with no positives
    (or all positives) in val fall back to `default`.
    """
    C = val_prob.shape[1]
    t = np.full(C, default, dtype=float)
    for c in range(C):
        y = val_true[:, c].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        p = val_prob[:, c]
        uniq = np.unique(p)
        if len(uniq) > 1:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            cand = np.concatenate(([0.0], mids, [1.0]))
        else:
            cand = np.array([0.0, float(uniq[0]), 1.0])
        preds = (p[None, :] >= cand[:, None]).astype(int)   # (n_cand, N)
        tp = (preds & y[None, :]).sum(axis=1)
        fp = preds.sum(axis=1) - tp
        fn = int(y.sum()) - tp
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
        t[c] = float(cand[int(np.argmax(f1))])
    return t


def _family_a(test_true, test_prob, t):
    pred = (test_prob >= t[None, :]).astype(int)
    tp = ((pred == 1) & (test_true == 1)).sum(0).astype(float)
    fp = ((pred == 1) & (test_true == 0)).sum(0).astype(float)
    fn = ((pred == 0) & (test_true == 1)).sum(0).astype(float)
    tn = ((pred == 0) & (test_true == 0)).sum(0).astype(float)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    sens = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
    return float(f1.mean()), float(sens.mean()), float(spec.mean())


def _family_b(test_true, test_prob):
    aucs, aps = [], []
    for c in range(test_prob.shape[1]):
        yt = test_true[:, c]
        if len(np.unique(yt)) < 2:          # all-pos or all-neg -> skip both
            continue
        aucs.append(roc_auc_score(yt, test_prob[:, c]))
        aps.append(average_precision_score(yt, test_prob[:, c]))
    return float(np.mean(aucs)), float(np.mean(aps))


def _family_c(test_true, test_prob):
    N, C = test_prob.shape
    # descending order; ties broken by class index (stable sort on -prob)
    order = np.argsort(-test_prob, axis=1, kind='stable')
    rec, tnr = [], {1: [], 2: [], 3: []}
    for i in range(N):
        ti = test_true[i]
        oi = order[i]
        n_i = int(ti.sum())
        n_neg = C - n_i
        if n_i >= 1:                                    # Recall@n
            top = oi[:n_i]
            rec.append(int(ti[top].sum()) / n_i)
        for j in (1, 2, 3):                             # TNR@(m-n)-j
            k = n_neg - j
            if k < 1:
                continue
            bot = oi[C - k:]                            # k lowest-prob classes
            tnr[j].append(int((ti[bot] == 0).sum()) / k)
    mean = lambda v: float(np.mean(v)) if v else float('nan')
    return mean(rec), {j: mean(v) for j, v in tnr.items()}


def compute_fiber_metrics(val_true, val_prob, test_true, test_prob):
    """Return the 9 metrics + the per-class threshold vector."""
    val_true = np.asarray(val_true).astype(int)
    test_true = np.asarray(test_true).astype(int)
    t = tune_thresholds(val_true, np.asarray(val_prob, float))
    mf1, sens, spec = _family_a(test_true, np.asarray(test_prob, float), t)
    auc, ap = _family_b(test_true, np.asarray(test_prob, float))
    rec_n, tnr = _family_c(test_true, np.asarray(test_prob, float))
    return {
        'macro_F1': mf1, 'macro_sensitivity': sens, 'macro_specificity': spec,
        'macro_AUROC': auc, 'macro_mAP': ap, 'Recall@n': rec_n,
        'TNR@(m-n)-1': tnr[1], 'TNR@(m-n)-2': tnr[2], 'TNR@(m-n)-3': tnr[3],
        'thresholds': t,
    }


# Metric display order / labels for reports.
METRIC_KEYS = [
    ('macro_F1', 'macro F1', 'A (per-class F1-opt thr)'),
    ('macro_sensitivity', 'macro sensitivity', 'A (per-class F1-opt thr)'),
    ('macro_specificity', 'macro specificity', 'A (per-class F1-opt thr)'),
    ('macro_AUROC', 'macro AUROC', 'B (threshold-free)'),
    ('macro_mAP', 'macro mAP', 'B (threshold-free)'),
    ('Recall@n', 'Recall@n', 'C (rank, per-image)'),
    ('TNR@(m-n)-1', 'TNR@(m-n)-1', 'C (rank, per-image)'),
    ('TNR@(m-n)-2', 'TNR@(m-n)-2', 'C (rank, per-image)'),
    ('TNR@(m-n)-3', 'TNR@(m-n)-3', 'C (rank, per-image)'),
]
