"""
Aggregate the multi-seed C-Tran runs into 95% confidence intervals under the
RetExpert protocol (metrics.txt) and write
performance/ctran_poly_constlr_bs32_<dataset>_summary.md, mirroring the format
of RetExpert's MURED_performance_summary.md.

Metrics per seed are computed from each run's saved best_preds.npz (test split,
best-val-F1 epoch) — deterministic, identical to reloading the checkpoint.

Usage: python summarize_ci.py [mured rfmid prism]
"""
import os
import sys

import numpy as np
from scipy import stats

from utils.retexpert_metrics import compute_metrics, recall_at_k

SEEDS = [1, 43, 44, 45, 46]
METRICS = [('f1', 'F1'), ('aupr', 'mAP'), ('auc', 'AUROC'),
           ('precision', 'Precision'), ('recall', 'Recall'),
           ('kappa', 'Kappa'), ('recall@5', 'Recall@5')]
RETEXPERT_MURED = dict(F1=0.6960, mAP=0.6378, AUROC=0.9302, Precision=0.6766,
                       Recall=0.7673, Kappa=0.6509, RecallAt5=0.9313)


def run_dir(dataset, seed):
    # RFMiD/PRISM seed-1 was the original default-named run; everything else
    # uses the per-seed name.
    if seed == 1 and dataset in ('rfmid', 'prism'):
        return f"results/{dataset}/ctran_poly_constlr_bs32"
    return f"results/{dataset}/ctran_poly_constlr_bs32_seed{seed}"


def seed_metrics(dataset, seed):
    d = run_dir(dataset, seed)
    npz = np.load(os.path.join(d, 'best_preds.npz'), allow_pickle=True)
    m = compute_metrics(npz['test_pred'], npz['test_true'])
    m['recall@5'] = recall_at_k(npz['test_pred'], npz['test_true'], k=5)
    return m


def ci95(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    mean = vals.mean()
    if n < 2:
        return mean, mean, mean
    se = vals.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return mean, mean - h, mean + h


def summarize(dataset):
    per_seed = {}
    for s in SEEDS:
        try:
            per_seed[s] = seed_metrics(dataset, s)
        except FileNotFoundError:
            print(f"  [warn] {dataset} seed {s}: best_preds.npz missing, skipping")
    seeds = sorted(per_seed)
    n = len(seeds)
    if n == 0:
        print(f"  [skip] {dataset}: no runs found")
        return None

    lines = [
        f"# C-Tran (DenseNet161, `ctran_poly_constlr_bs32`) on {dataset.upper()} "
        "— Multi-seed Summary",
        "",
        f"Seeds: {seeds} (n={n}). Config: PolyLoss, constant LR 1e-5, effective "
        "batch 32, LP-ROS 10%, LMT, 384px, 80 epochs; best epoch by validation "
        "F1. Metrics: RetExpert protocol (`metrics.txt`), TEST split, all classes.",
        "",
        "## Mean metrics (95% CI)",
        "",
        "| Metric | Mean (95% CI) |",
        "|---|---|",
    ]
    means = {}
    for key, lbl in METRICS:
        vals = [per_seed[s][key] for s in seeds]
        mean, lo, hi = ci95(vals)
        means[lbl] = mean
        lines.append(f"| {lbl} | {mean:.4f} (95% CI [{lo:.4f}, {hi:.4f}]) |")

    lines += ["", "## Per-seed results", "",
              "| Seed | " + " | ".join(l for _, l in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
    for s in seeds:
        row = " | ".join(f"{per_seed[s][k]:.4f}" for k, _ in METRICS)
        lines.append(f"| {s} | {row} |")

    if dataset == 'mured':
        lines += ["", "## vs RetExpert (MuReD, 200-epoch 5-seed mean)", "",
                  "| Metric | C-Tran mean | RetExpert |", "|---|---|---|"]
        for key, lbl in METRICS:
            rk = 'RecallAt5' if lbl == 'Recall@5' else lbl
            lines.append(f"| {lbl} | {means[lbl]:.4f} | {RETEXPERT_MURED[rk]:.4f} |")
    lines.append("")

    os.makedirs('performance', exist_ok=True)
    path = f"performance/ctran_poly_constlr_bs32_{dataset}_summary.md"
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}  (n={n})  F1={means['F1']:.4f} AUROC={means['AUROC']:.4f}")
    return means


def main():
    datasets = sys.argv[1:] or ['mured', 'rfmid', 'prism']
    for ds in datasets:
        print(f"[{ds}]")
        summarize(ds)


if __name__ == '__main__':
    main()
