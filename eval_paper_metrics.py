"""Reproduction vs claimed, on the metrics each paper actually reports (MuReD test).

C-Tran    : ML_F1 / Bin_AUC / Bin_F1 as Rodriguez et al. define them in Sec. V.A --
            disease classes only (NORMAL excluded) for ML_*, and the NORMAL column
            itself for Bin_*. Reused from C-Tran/utils/mured_metrics.py, whose
            docstring records that it was validated against the paper's Table XII.
RetExpert : F1 and Kappa as the RetExpert engine reports them -- per-sample F1 and
            Cohen's kappa on the flattened matrix (average='samples', threshold 0.5).

Mean over seeds 42-46 with a t-based 95% CI, from the cached test probabilities.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'C-Tran'))
from utils.mured_metrics import compute_mured_metrics
from utils.retexpert_metrics import compute_metrics

PROBS = os.environ['PROBS_DIR']
RUNS = os.environ['RUNS_DIR']
SEEDS = (42, 43, 44, 45, 46)

CLAIMED = {                       # values quoted in each paper
    'C-Tran ML_F1':    0.5730,
    'C-Tran Bin_AUC':  0.9760,
    'C-Tran Bin_F1':   0.8240,
    'RETExpert F1':    0.7301,
    'RETExpert Kappa': 0.70844,
}


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
