"""Fiber evaluation protocol — the 9 metrics defined in metrics.txt.

Families:
  A. macro, threshold-based : macro F1, macro sensitivity, macro specificity
     (per-class F1-optimal threshold t_c tuned on validation, applied to test)
  B. macro, threshold-free  : macro AUROC, macro mAP
  C. per-image, rank-based  : Recall@n, TNR@(m-n)-1, TNR@(m-n)-2, TNR@(m-n)-3

Pure numpy/sklearn — no torch, no GPU. All metrics computed once over the full
concatenated (N, C) prediction/target matrices.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_KEYS = [
    "macro_f1", "macro_sensitivity", "macro_specificity",   # A
    "macro_auroc", "macro_map",                             # B
    "recall_at_n", "tnr_mn_1", "tnr_mn_2", "tnr_mn_3",      # C
]


# ---------- Family A: per-class F1-optimal thresholds (tuned on validation) ----------
def _best_threshold_for_class(val_probs_c, val_targ_c):
    """t_c = argmax over candidate cut-points of binary F1 on validation.

    Candidates: midpoints between consecutive sorted unique validation probs of
    the class, plus sentinels {0, 1}. No positives in validation -> fallback 0.5.
    """
    if val_targ_c.sum() == 0:
        return 0.5
    u = np.unique(val_probs_c)
    mids = (u[:-1] + u[1:]) / 2.0 if u.size >= 2 else np.array([])
    cands = np.concatenate(([0.0], mids, [1.0]))
    P = int(val_targ_c.sum())
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        pred = val_probs_c >= t
        tp = int(np.sum(pred & (val_targ_c == 1)))
        fp = int(np.sum(pred & (val_targ_c == 0)))
        denom = 2 * tp + fp + (P - tp)  # 2TP + FP + FN, with FN = P - TP
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def tune_thresholds(val_probs, val_targets):
    """Return the per-class threshold vector t_c (length C)."""
    C = val_probs.shape[1]
    return np.array([_best_threshold_for_class(val_probs[:, c], val_targets[:, c])
                     for c in range(C)])


def _family_A(test_probs, test_targets, thresholds):
    C = test_probs.shape[1]
    f1s, sens, spec = [], [], []
    for c in range(C):
        pred = test_probs[:, c] >= thresholds[c]
        yt = test_targets[:, c] == 1
        tp = int(np.sum(pred & yt))
        fp = int(np.sum(pred & ~yt))
        fn = int(np.sum(~pred & yt))
        tn = int(np.sum(~pred & ~yt))
        f1s.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
        sens.append((tp / (tp + fn)) if (tp + fn) else 0.0)
        spec.append((tn / (tn + fp)) if (tn + fp) else 0.0)
    return (float(np.mean(f1s)), float(np.mean(sens)), float(np.mean(spec)),
            {"f1": f1s, "sensitivity": sens, "specificity": spec})


# ---------- Family B: threshold-free macro AUROC / mAP ----------
def _family_B(test_probs, test_targets):
    aurocs, aps = [], []
    for c in range(test_probs.shape[1]):
        yt = test_targets[:, c]
        if len(np.unique(yt)) < 2:      # all-pos or all-neg in test -> skip
            continue
        aurocs.append(roc_auc_score(yt, test_probs[:, c]))
        aps.append(average_precision_score(yt, test_probs[:, c]))
    return float(np.mean(aurocs)), float(np.mean(aps)), len(aurocs)


# ---------- Family C: per-image rank-based ----------
def _family_C(test_probs, test_targets):
    N, m = test_probs.shape
    recall_n = []
    tnr = {1: [], 2: [], 3: []}
    for i in range(N):
        p = test_probs[i]
        t = test_targets[i]
        n_i = int(t.sum())
        # Recall@n : top n_i by descending prob (stable, ties by class index)
        if n_i > 0:
            top = np.argsort(-p, kind="stable")[:n_i]
            recall_n.append(int(t[top].sum()) / n_i)
        # TNR@(m-n)-j : bottom (neg - j) by ascending prob
        neg = m - n_i
        asc = np.argsort(p, kind="stable")
        for j in (1, 2, 3):
            k = neg - j
            if k >= 1:
                bot = asc[:k]
                tnr[j].append(int(np.sum(t[bot] == 0)) / k)
    return (
        float(np.mean(recall_n)) if recall_n else float("nan"),
        {j: (float(np.mean(v)) if v else float("nan")) for j, v in tnr.items()},
        len(recall_n),
    )


def compute_all(val_probs, val_targets, test_probs, test_targets):
    """Tune thresholds on validation, compute all 9 metrics on test."""
    t_c = tune_thresholds(val_probs, val_targets)
    f1, sens, spec, per_class = _family_A(test_probs, test_targets, t_c)
    auroc, mAP, n_eval = _family_B(test_probs, test_targets)
    r_at_n, tnr, n_img_r = _family_C(test_probs, test_targets)
    return {
        "macro_f1": f1,
        "macro_sensitivity": sens,
        "macro_specificity": spec,
        "macro_auroc": auroc,
        "macro_map": mAP,
        "recall_at_n": r_at_n,
        "tnr_mn_1": tnr[1],
        "tnr_mn_2": tnr[2],
        "tnr_mn_3": tnr[3],
        "thresholds": t_c.tolist(),
        "per_class_A": per_class,
        "n_classes": int(test_probs.shape[1]),
        "n_classes_evaluable_B": n_eval,
        "n_images_recall_at_n": n_img_r,
        "n_images": int(test_probs.shape[0]),
    }
