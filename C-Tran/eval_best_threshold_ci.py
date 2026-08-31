"""
Best-performance-threshold performance report with 95% CIs for
ctran_poly_constlr_bs32, per the updated metrics.txt protocol:

  F1 / Precision / Recall  -> single GLOBAL threshold that maximizes per-sample
                              F1, tuned on VALIDATION, applied to TEST.
  AUROC / mAP / Recall@5   -> threshold-free / rank-based (unchanged).
  Kappa / Accuracy         -> fixed 0.5 (unchanged).

Aggregated over seeds {1,43,44,45,46}. Writes
performance/ctran_poly_constlr_bs32_best_threshold_ci.md.
"""
import os
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.retexpert_metrics import compute_metrics, recall_at_k

SEEDS = [1, 43, 44, 45, 46]
GRID = np.arange(0.01, 1.00, 0.01)
DATASETS = ['mured', 'rfmid', 'prism']
METRICS = [('f1', 'F1'), ('aupr', 'mAP'), ('auc', 'AUROC'),
           ('precision', 'Precision'), ('recall', 'Recall'),
           ('kappa', 'Kappa'), ('recall@5', 'Recall@5')]


def run_dir(dataset, seed):
    if seed == 1 and dataset in ('rfmid', 'prism'):
        return f"results/{dataset}/ctran_poly_constlr_bs32"
    return f"results/{dataset}/ctran_poly_constlr_bs32_seed{seed}"


def best_f1_threshold(y_true, y_prob):
    """Single global threshold maximizing per-sample F1 (argmax over grid)."""
    best_t, best_f1 = 0.5, -1.0
    for t in GRID:
        f1 = f1_score(y_true, (y_prob >= t).astype(int),
                      average='samples', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def ci95(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    h = stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return mean, mean - h, mean + h


def seed_metrics(dataset, seed):
    z = np.load(os.path.join(run_dir(dataset, seed), 'best_preds.npz'),
                allow_pickle=True)
    vt, vp = z['val_true'], z['val_pred']
    tt, tp = z['test_true'], z['test_pred']

    t_star = best_f1_threshold(vt, vp)                 # tuned on validation
    preds = (tp >= t_star).astype(int)                 # applied to test

    m = compute_metrics(tp, tt)                        # auc/aupr/kappa/acc/hamming
    m['recall@5'] = recall_at_k(tp, tt, k=5)
    # override P/R/F1 with the best-F1-threshold values
    m['f1'] = float(f1_score(tt, preds, average='samples', zero_division=0))
    m['precision'] = float(precision_score(tt, preds, average='samples', zero_division=0))
    m['recall'] = float(recall_score(tt, preds, average='samples', zero_division=0))
    m['threshold'] = t_star
    return m


def summarize(dataset, lines):
    per = {s: seed_metrics(dataset, s) for s in SEEDS}
    seeds = sorted(per)
    n = len(seeds)
    tmean, tlo, thi = ci95([per[s]['threshold'] for s in seeds])

    lines += [f"## {dataset.upper()}  (n={n} seeds)", "",
              f"Val-tuned global F1-threshold: mean {tmean:.3f} "
              f"(range {min(per[s]['threshold'] for s in seeds):.2f}"
              f"-{max(per[s]['threshold'] for s in seeds):.2f}).", "",
              "| Metric | Mean (95% CI) | Threshold |",
              "|---|---|---|"]
    means = {}
    for key, lbl in METRICS:
        mean, lo, hi = ci95([per[s][key] for s in seeds])
        means[lbl] = mean
        thr = "best-F1 (val)" if lbl in ('F1', 'Precision', 'Recall') else \
              ("0.5" if lbl == 'Kappa' else "n/a")
        lines.append(f"| {lbl} | {mean:.4f} ([{lo:.4f}, {hi:.4f}]) | {thr} |")
    lines += ["", "### Per-seed (test)", "",
              "| Seed | thr* | " + " | ".join(l for _, l in METRICS) + " |",
              "|---|---|" + "---|" * len(METRICS)]
    for s in seeds:
        row = " | ".join(f"{per[s][k]:.4f}" for k, _ in METRICS)
        lines.append(f"| {s} | {per[s]['threshold']:.2f} | {row} |")
    lines.append("")
    print(f"[{dataset}] F1={means['F1']:.4f} P={means['Precision']:.4f} "
          f"R={means['Recall']:.4f} (thr~{tmean:.2f})  AUROC={means['AUROC']:.4f}")
    return means


def main():
    lines = [
        "# C-Tran (`ctran_poly_constlr_bs32`) — best-threshold performance, 95% CI",
        "",
        "Metrics per the updated `metrics.txt`: **F1 / Precision / Recall at the "
        "single global threshold that maximizes per-sample F1, tuned on the "
        "validation split and applied to the test split**; AUROC / mAP / Recall@5 "
        "threshold-free; Kappa at 0.5. Aggregated over seeds {1,43,44,45,46} "
        "(t-based 95% CI). TEST split, all classes.",
        "",
    ]
    for ds in DATASETS:
        summarize(ds, lines)
    os.makedirs('performance', exist_ok=True)
    out = 'performance/ctran_poly_constlr_bs32_best_threshold_ci.md'
    with open(out, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
