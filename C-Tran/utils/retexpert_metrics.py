"""
RetExpert evaluation protocol, replicated verbatim so C-Tran numbers are
directly comparable to the RetExpert MuReD results.

Source of truth: /home/cousin/research/RetExpert/utils/engine.py
(_safe_macro_score, compute_metrics) and
/home/cousin/research/RetExpert/scripts/compute_recall_at_5.py (recall_at_k),
as specified in ./metrics.txt.

Key protocol points (differ from the MuReD paper's own metrics):
  - probs = sigmoid(logits), independently per class.
  - Fixed 0.5 threshold everywhere; no per-class tuning/calibration.
  - Precision/Recall/F1: average='samples' (per-sample), zero_division=0.
  - Accuracy: sklearn accuracy_score on the (N,C) matrix = SUBSET accuracy.
  - AUROC / mAP: per-class macro, skipping degenerate columns.
  - Kappa: cohen_kappa_score on FLATTENED (N*C) arrays, thresholded.
  - Recall@5: rank-based; samples with zero positives excluded.
  - Computed over ALL classes (NORMAL included) on the TEST split.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, average_precision_score, hamming_loss
)


def _safe_macro_score(score_fn, targets, probs):
    """Per-class macro average that skips degenerate columns."""
    scores = []
    for c in range(targets.shape[1]):
        yt = targets[:, c]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(score_fn(yt, probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(probs, targets, threshold=0.5, average='samples'):
    preds = (probs >= threshold).astype(int)
    targets = targets.astype(int)

    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)

    auc = _safe_macro_score(roc_auc_score, targets, probs)
    aupr = _safe_macro_score(average_precision_score, targets, probs)

    kappa = cohen_kappa_score(targets.flatten(), preds.flatten())
    hamming = hamming_loss(targets, preds)

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "aupr": aupr,
        "kappa": float(kappa),
        "hamming": float(hamming),
    }


def recall_at_k(probs, targets, k=5):
    """Per-sample recall@k, averaged over samples with >=1 positive label."""
    n_samples, n_classes = probs.shape
    k = min(k, n_classes)
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    scores = []
    for i in range(n_samples):
        pos = np.where(targets[i] == 1)[0]
        if len(pos) == 0:
            continue
        hits = np.intersect1d(topk_idx[i], pos)
        scores.append(len(hits) / len(pos))
    return float(np.mean(scores)) if scores else 0.0
