"""
Generate the metrics.txt (Fiber 9-metric protocol) per-dataset performance
reports for ctran_poly_constlr_bs32, aggregated over seeds {1,43,44,45,46}
(5-seed 95% CI). Per-class F1-optimal threshold vectors -> auditable CSV sidecars.

FLAG-EXCLUSIVE class-set standard (agreed across the 4 compared models):
  - MuReD : 20 classes, UNCHANGED (its NORMAL is a native label, kept).
  - RFMiD : 28 classes, drop `Disease_Risk` (screening meta-flag).
  - PRISM : 31 classes, drop `NORMAL` (screening meta-flag).
The flag column is dropped BY NAME from BOTH probs and targets *before*
compute_fiber_metrics, so threshold tuning and all 9 metrics run on the reduced
set. (We drop the column from the cached full-class prob matrices in
best_preds.npz, which is identical to re-inferring then dropping it.)

Writes performance/ctran_poly_constlr_bs32_{rfmid,prism}.md (+ threshold CSVs).
"""
import os
import csv
from datetime import date
import numpy as np
from scipy import stats

from utils.fiber_metrics import compute_fiber_metrics, METRIC_KEYS

SEEDS = [1, 43, 44, 45, 46]
DROP_FLAG = {'mured': None, 'rfmid': 'Disease_Risk', 'prism': 'NORMAL'}


def run_dir(dataset, seed):
    if seed == 1 and dataset in ('rfmid', 'prism'):
        return f"results/{dataset}/ctran_poly_constlr_bs32"
    return f"results/{dataset}/ctran_poly_constlr_bs32_seed{seed}"


def ci95(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    h = stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return mean, mean - h, mean + h


def load_seed(dataset, seed, drop_flag):
    """Return (val_true, val_pred, test_true, test_pred, label_cols) with the
    named flag column removed from probs AND targets (if present)."""
    z = np.load(os.path.join(run_dir(dataset, seed), 'best_preds.npz'),
                allow_pickle=True)
    lc = [str(c) for c in z['label_cols']]
    vt, vp, tt, tp = z['val_true'], z['val_pred'], z['test_true'], z['test_pred']
    if drop_flag is not None:
        assert drop_flag in lc, f"{dataset}: flag '{drop_flag}' not in {lc}"
        fi = lc.index(drop_flag)
        keep = [i for i in range(len(lc)) if i != fi]
        lc = [lc[i] for i in keep]
        vt, vp, tt, tp = vt[:, keep], vp[:, keep], tt[:, keep], tp[:, keep]
    return vt, vp, tt, tp, lc


def report(dataset):
    flag = DROP_FLAG[dataset]
    per, thr_rows, label_cols = {}, {}, None
    n_test = n_classes = None
    for s in SEEDS:
        vt, vp, tt, tp, label_cols = load_seed(dataset, s, flag)
        n_test, n_classes = tt.shape
        per[s] = compute_fiber_metrics(vt, vp, tt, tp)
        thr_rows[s] = per[s]['thresholds']
    seeds = sorted(per)

    os.makedirs('performance/thresholds', exist_ok=True)
    csv_path = f"performance/thresholds/ctran_poly_constlr_bs32_{dataset}_thresholds.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed'] + label_cols)
        for s in seeds:
            w.writerow([s] + [f"{v:.4f}" for v in thr_rows[s]])

    thr_all = np.array([thr_rows[s] for s in seeds])
    flag_note = (f" ({DROP_FLAG[dataset]} screening flag dropped — flag-exclusive "
                 "standard)") if flag else ""
    L = [
        f"# C-Tran (DenseNet161, `ctran_poly_constlr_bs32`) on {dataset.upper()} "
        "— Test Performance (Fiber protocol)",
        "",
        f"Generated {date.today().isoformat()}. Metrics per `metrics.txt` "
        "(Fiber 9-metric protocol): **Family A** (macro F1 / sensitivity / "
        "specificity) at **per-class F1-optimal thresholds tuned on validation**, "
        "applied to test; **Family B** (macro AUROC / mAP) threshold-free; "
        "**Family C** (Recall@n, TNR@(m-n)-1/-2/-3) per-image rank-based. "
        f"Seeds {seeds} (n={len(seeds)}), t-based 95% CI, TEST split "
        f"({n_test} images, **{n_classes} classes**{flag_note}).",
        "",
        "## Configuration",
        "- Model: C-Tran (3 transformer layers, 4 heads, dropout 0.1, hidden "
        "2208) + DenseNet161 backbone",
        "- 384x384 + FOV crop; PolyLoss (eps=1); Adam lr 1e-5 (constant); "
        "LP-ROS 10%; LMT training / no-prior inference; effective batch 32 "
        "(16x2 grad-accum); 80 epochs; checkpoint per seed by validation F1",
        f"- Class set: **{n_classes} classes**"
        + (f" — `{flag}` dropped from the model's native {n_classes + 1}-output "
           "head (by name, from probs & targets) before metric computation; no "
           "retraining." if flag else " (native label set, unchanged)"),
        f"- Per-class F1-optimal thresholds (t_c) tuned on validation — full "
        f"per-seed vectors: `thresholds/ctran_poly_constlr_bs32_{dataset}_thresholds.csv` "
        f"(across classes & seeds: mean {thr_all.mean():.3f}, "
        f"range {thr_all.min():.2f}-{thr_all.max():.2f})",
        "",
        "## Test metrics — 5-seed mean (95% CI)",
        "",
        "| # | Metric | Family | Mean (95% CI) |",
        "|---|---|---|---|",
    ]
    for idx, (key, lbl, fam) in enumerate(METRIC_KEYS, 1):
        m, lo, hi = ci95([per[s][key] for s in seeds])
        L.append(f"| {idx} | {lbl} | {fam} | {m:.4f} ([{lo:.4f}, {hi:.4f}]) |")

    L += ["", "## Per-seed (test)", "",
          "| Seed | macro F1 | sens | spec | AUROC | mAP | Recall@n | "
          "TNR@(m-n)-1 | -2 | -3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        d = per[s]
        L.append(f"| {s} | {d['macro_F1']:.4f} | {d['macro_sensitivity']:.4f} "
                 f"| {d['macro_specificity']:.4f} | {d['macro_AUROC']:.4f} "
                 f"| {d['macro_mAP']:.4f} | {d['Recall@n']:.4f} "
                 f"| {d['TNR@(m-n)-1']:.4f} | {d['TNR@(m-n)-2']:.4f} "
                 f"| {d['TNR@(m-n)-3']:.4f} |")
    L.append("")

    out = f"performance/ctran_poly_constlr_bs32_{dataset}.md"
    with open(out, 'w') as f:
        f.write("\n".join(L))
    f1m, _, _ = ci95([per[s]['macro_F1'] for s in seeds])
    print(f"wrote {out}  ({n_classes} cls) macroF1={f1m:.4f}  (+ {csv_path})")


def sanity_before_after():
    """Print full-class vs flag-exclusive Recall@n (+ macro F1) for RFMiD/PRISM."""
    print("\n=== SANITY: full class-set  vs  flag-exclusive (5-seed mean) ===")
    for ds in ('rfmid', 'prism'):
        flag = DROP_FLAG[ds]
        full_rec, red_rec, full_f1, red_f1 = [], [], [], []
        nc_full = nc_red = None
        for s in SEEDS:
            vt, vp, tt, tp, lc = load_seed(ds, s, None)          # full
            nc_full = tt.shape[1]
            mf = compute_fiber_metrics(vt, vp, tt, tp)
            vt2, vp2, tt2, tp2, lc2 = load_seed(ds, s, flag)     # reduced
            nc_red = tt2.shape[1]
            mr = compute_fiber_metrics(vt2, vp2, tt2, tp2)
            full_rec.append(mf['Recall@n']); red_rec.append(mr['Recall@n'])
            full_f1.append(mf['macro_F1']);  red_f1.append(mr['macro_F1'])
        fr, rr = np.mean(full_rec), np.mean(red_rec)
        ff, rf = np.mean(full_f1), np.mean(red_f1)
        print(f"  {ds.upper()}  drop '{flag}':")
        print(f"    Recall@n : {fr:.4f} ({nc_full} cls)  ->  {rr:.4f} ({nc_red} cls)   "
              f"delta {rr - fr:+.4f}")
        print(f"    macro F1 : {ff:.4f} ({nc_full} cls)  ->  {rf:.4f} ({nc_red} cls)   "
              f"delta {rf - ff:+.4f}")


def main():
    for ds in ('rfmid', 'prism'):   # MuReD left untouched
        report(ds)
    sanity_before_after()


if __name__ == '__main__':
    main()
