"""Seven-metric evaluation protocol (F1, mAP, AUROC, Precision, Recall, Kappa, Recall@5).

Implements the protocol specified in metrics.txt (RetExpert's utils/engine.py +
scripts/compute_recall_at_5.py) so numbers from this codebase are directly
comparable to RetExpert's.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score, f1_score,
    hamming_loss, precision_score, recall_score, roc_auc_score,
)

THRESHOLD = 0.5


def _safe_macro_score(fn, targets, probs):
    """Macro-average fn over classes, skipping degenerate (single-value) columns."""
    scores, n_skipped = [], 0
    for c in range(targets.shape[1]):
        if len(np.unique(targets[:, c])) < 2:
            n_skipped += 1
            continue
        scores.append(fn(targets[:, c], probs[:, c]))
    return float(np.mean(scores)), len(scores), n_skipped


def per_sample_f1(probs, targets, threshold):
    preds = (probs >= threshold).astype(int)
    return float(f1_score(targets, preds, average="samples", zero_division=0))


def find_best_threshold(probs, targets, lo=0.01, hi=0.99, step=0.01):
    """Single global threshold maximizing per-sample F1 (spec: tuned on validation)."""
    ts = np.round(np.arange(lo, hi + step / 2, step), 4)
    f1s = [per_sample_f1(probs, targets, t) for t in ts]
    best = int(np.argmax(f1s))
    return float(ts[best]), float(f1s[best])


def recall_at_k(probs, targets, k=5):
    """Per-sample rank-based recall: |top-k ∩ true| / |true|, excluding empty-label rows."""
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    scores = []
    for i in range(probs.shape[0]):
        n_true = int(targets[i].sum())
        if n_true == 0:
            continue
        hits = int(targets[i, topk[i]].sum())
        scores.append(hits / n_true)
    return float(np.mean(scores)), len(scores)


def compute_metrics(probs, targets, pr_threshold, fixed_threshold=THRESHOLD):
    """Compute the 7-metric set.

    pr_threshold: global threshold for P/R/F1 (F1-optimal, tuned on validation).
    fixed_threshold: 0.5 cutoff for Kappa and subset Accuracy (per metrics.txt regime b).
    """
    pr_preds = (probs >= pr_threshold).astype(int)
    fixed_preds = (probs >= fixed_threshold).astype(int)

    auroc, n_auroc, n_skip = _safe_macro_score(roc_auc_score, targets, probs)
    mAP, _, _ = _safe_macro_score(average_precision_score, targets, probs)
    r_at_5, n_r5 = recall_at_k(probs, targets, k=5)

    return {
        "f1": float(f1_score(targets, pr_preds, average="samples", zero_division=0)),
        "mAP": mAP,
        "auroc": auroc,
        "precision": float(precision_score(targets, pr_preds, average="samples", zero_division=0)),
        "recall": float(recall_score(targets, pr_preds, average="samples", zero_division=0)),
        "kappa": float(cohen_kappa_score(targets.flatten(), fixed_preds.flatten())),
        "recall_at_5": r_at_5,
        "accuracy_subset": float(accuracy_score(targets, fixed_preds)),
        "hamming_loss": float(hamming_loss(targets, fixed_preds)),
        "pr_threshold": float(pr_threshold),
        "n_samples": int(targets.shape[0]),
        "n_classes": int(targets.shape[1]),
        "n_classes_scored": n_auroc,
        "n_classes_skipped": n_skip,
        "n_samples_recall_at_5": n_r5,
    }


def per_class_table(probs, targets, label_cols, threshold=THRESHOLD):
    out = {}
    preds = (probs >= threshold).astype(int)
    for i, col in enumerate(label_cols):
        yt, yp, pr = targets[:, i], probs[:, i], preds[:, i]
        if len(np.unique(yt)) < 2:
            out[col] = {"auc": None, "ap": None, "n_pos": int(yt.sum())}
            continue
        out[col] = {
            "auc": round(float(roc_auc_score(yt, yp)), 4),
            "ap": round(float(average_precision_score(yt, yp)), 4),
            "f1": round(float(f1_score(yt, pr, zero_division=0)), 4),
            "n_pos": int(yt.sum()),
            "n_pred_pos": int(pr.sum()),
        }
    return out
