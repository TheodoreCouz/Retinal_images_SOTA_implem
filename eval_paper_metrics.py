"""Reproduction vs claimed, on the metrics each paper actually reports.

C-Tran    (MuReD test) : ML_F1 / Bin_AUC / Bin_F1 as Rodriguez et al. define them
            in Sec. V.A -- disease classes only (NORMAL excluded) for ML_*, and
            the NORMAL column itself for Bin_*. Reused from
            C-Tran/utils/mured_metrics.py, whose docstring records that it was
            validated against the paper's Table XII.
RetExpert (MuReD test) : F1 and Kappa as the RetExpert engine reports them --
            per-sample F1 and Cohen's kappa on the flattened matrix
            (average='samples', threshold 0.5).
RIADD/KAMATALAB (RFMiD test) : MD Avg / SC2 F1 / Specificity as the challenge
            paper's own Table 7 defines them -- MD Avg = 1/2*(macro AUC + macro
            AP) over the 28 disease columns (threshold-free); SC2 F1 / SC2
            Specificity = macro F1 / specificity over the same 28 columns at
            per-class thresholds tuned on validation (metrics9.tune_thresholds,
            the same tuned-threshold protocol as eval_riadd_paths.py's ablation
            row -- upstream never specifies a threshold, so this is one
            reasonable choice, not a recovered one). Uses the TTA-corrected
            3-path blend (riadd3_blend_*, per-path TTA 5/3/3 per Appendix A.1).

Mean over seeds 42-46 with a t-based 95% CI, from the cached test probabilities.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C-Tran'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Pytorch-RIADD'))
from utils.mured_metrics import compute_mured_metrics
from utils.retexpert_metrics import compute_metrics
from metrics9 import tune_thresholds

PROBS = os.environ['PROBS_DIR']
RUNS = os.environ['RUNS_DIR']
SEEDS = (42, 43, 44, 45, 46)

CLAIMED = {                       # values quoted in each paper
    'C-Tran ML_F1':          0.5730,
    'C-Tran Bin_AUC':        0.9760,
    'C-Tran Bin_F1':         0.8240,
    'RETExpert F1':          0.7301,
    'RETExpert Kappa':       0.70844,
    'RIADD MD Avg':          0.7821,
    'RIADD SC2 F1':          0.5546,
    'RIADD Specificity':     0.9957,
}


def macro(fn, p, t):
    return float(np.mean([fn(t[:, c], p[:, c]) for c in range(t.shape[1])
                          if len(np.unique(t[:, c])) > 1]))


def ci95(vals):
    v = np.asarray(vals, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


rows = {}

# --- C-Tran: the paper's own ML_/Bin_ protocol -----------------------------
ct = {'ML_F1': [], 'Bin_AUC': [], 'Bin_F1': []}
for s in SEEDS:
    z = np.load(f'{PROBS}/ctran_mured_seed{s}.npz', allow_pickle=True)
    cols = [str(c) for c in z['label_cols']]
    m = compute_mured_metrics(z['test_true'], z['test_pred'], cols)
    for k in ct:
        ct[k].append(m[k])
for k, v in ct.items():
    rows[f'C-Tran {k}'] = v

# --- RetExpert: the paper's primary metrics --------------------------------
rx = {'F1': [], 'Kappa': []}
for s in SEEDS:
    z = np.load(f'{PROBS}/retexpert_mured_seed{s}.npz')
    m = compute_metrics(z['test_pred'], z['test_true'])   # average='samples'
    rx['F1'].append(m['f1'])
    rx['Kappa'].append(m['kappa'])
for k, v in rx.items():
    rows[f'RETExpert {k}'] = v

# --- RIADD/KAMATALAB: the paper's own MD Avg / SC2 F1 / Specificity --------
from sklearn.metrics import average_precision_score, roc_auc_score

rd = {'MD Avg': [], 'SC2 F1': [], 'Specificity': []}
for s in SEEDS:
    te = np.load(f'{PROBS}/riadd3_blend_test_seed{s}.npz')
    va = np.load(f'{PROBS}/riadd3_blend_val_seed{s}.npz')
    p, t = te['probs'], te['targets']
    thr = tune_thresholds(va['probs'], va['targets'])
    pred = (p[:, 1:] >= thr[None, 1:]).astype(int)
    y = t[:, 1:].astype(int)
    md = 0.5 * (macro(roc_auc_score, p[:, 1:], t[:, 1:])
                + macro(average_precision_score, p[:, 1:], t[:, 1:]))
    f1s, specs = [], []
    for c in range(y.shape[1]):
        tp = int(((pred[:, c] == 1) & (y[:, c] == 1)).sum())
        fp = int(((pred[:, c] == 1) & (y[:, c] == 0)).sum())
        fn = int(((pred[:, c] == 0) & (y[:, c] == 1)).sum())
        tn = int(((pred[:, c] == 0) & (y[:, c] == 0)).sum())
        f1s.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0)
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    rd['MD Avg'].append(md)
    rd['SC2 F1'].append(float(np.mean(f1s)))
    rd['Specificity'].append(float(np.mean(specs)))
for k, v in rd.items():
    rows[f'RIADD {k}'] = v

# --- table -----------------------------------------------------------------
print(f"{'Metric':22s} {'Reproduction (95% CI)':30s} {'Claimed':>9s} {'Gap':>8s}")
print('-' * 72)
for k, v in rows.items():
    mean, h = ci95(v)
    c = CLAIMED[k]
    print(f"{k:22s} {mean*100:6.2f}%  [{(mean-h)*100:5.2f}, {(mean+h)*100:5.2f}]      "
          f"{c*100:7.2f}%  {(mean-c)*100:+7.2f}")

print('\nper-seed')
for k, v in rows.items():
    print(f"  {k:22s} " + '  '.join(f'{x*100:6.2f}' for x in v))

# --- cross-check: RetExpert's own training-time JSONs ----------------------
js = []
for s in SEEDS:
    p = f'{RUNS}/retexpert/performance/MuReD_retexpert_seed{s}_ep200_test_performance.json'
    js.append(json.load(open(p))['test_metrics']['f1'])
print(f"\ncross-check RETExpert F1 recomputed {np.mean(rows['RETExpert F1'])*100:.2f}% "
      f"vs training-time JSONs {np.mean(js)*100:.2f}%")
