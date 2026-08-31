"""
Generate a paper-comparison report from a finished C-Tran MuReD run.

Reads best_preds.npz (saved by train_mured.py at the best-val epoch) and prints:
  - Table XI-style headline comparison (our model vs paper's proposed model)
  - Table XII-style per-class breakdown (TEST split)

Usage: python report_mured.py results/mured/<run_name>
"""
import sys
import os
import numpy as np

from utils.mured_metrics import compute_mured_metrics

# Paper Table XI, "Proposed model" row.
PAPER = dict(ML_F1=0.573, ML_mAP=0.685, ML_AUC=0.962,
             Bin_AUC=0.976, Bin_F1=0.824, Model_Score=0.900)
# Paper Table XII per-class (Precision, Recall, F1, AUC).
PAPER_PER_CLASS = {
    'DR': (0.859, 0.859, 0.859, 0.962), 'NORMAL': (0.865, 0.786, 0.824, 0.976),
    'MH': (0.875, 0.618, 0.724, 0.962), 'ODC': (0.661, 0.750, 0.703, 0.966),
    'TSLN': (0.800, 0.774, 0.787, 0.989), 'ARMD': (0.800, 0.500, 0.615, 0.965),
    'DN': (0.708, 0.531, 0.615, 0.938), 'MYA': (0.810, 0.944, 0.872, 0.997),
    'BRVO': (0.929, 0.813, 0.867, 0.994), 'ODP': (0.000, 0.000, 0.000, 0.870),
    'CRVO': (0.600, 0.545, 0.571, 0.981), 'CNV': (0.889, 0.667, 0.762, 0.992),
    'RS': (1.000, 0.545, 0.706, 0.971), 'ODE': (0.833, 0.909, 0.870, 0.971),
    'LS': (0.500, 0.556, 0.526, 0.990), 'CSR': (0.444, 0.571, 0.500, 0.981),
    'HTR': (0.000, 0.000, 0.000, 0.911), 'ASR': (0.000, 0.000, 0.000, 0.971),
    'CRS': (0.400, 0.333, 0.364, 0.988), 'OTHER': (0.587, 0.519, 0.551, 0.851),
}


def headline_table(ours, split):
    keys = ['ML_F1', 'ML_mAP', 'ML_AUC', 'Bin_AUC', 'Bin_F1', 'Model_Score']
    print(f"\n{'Metric':<13}{'Ours ('+split+')':>16}{'Paper':>10}{'Delta':>10}")
    print("-" * 49)
    for k in keys:
        d = ours[k] - PAPER[k]
        print(f"{k:<13}{ours[k]:>16.3f}{PAPER[k]:>10.3f}{d:>+10.3f}")


def per_class_table(ours):
    print(f"\n{'class':<8}{'n':>4}  {'Prec':>15}  {'Recall':>15}  {'F1':>15}  {'AUC':>15}")
    print(f"{'':<8}{'':>4}  {'ours / paper':>15}  {'ours / paper':>15}  "
          f"{'ours / paper':>15}  {'ours / paper':>15}")
    print("-" * 88)
    for c, d in ours['per_class'].items():
        p = PAPER_PER_CLASS.get(c, (None,) * 4)
        auc = d['auc'] if d['auc'] is not None else float('nan')

        def cell(o, pv):
            pv = 'n/a' if pv is None else f"{pv:.3f}"
            return f"{o:.3f} / {pv}"
        print(f"{c:<8}{d['n_pos']:>4}  {cell(d['precision'],p[0]):>15}  "
              f"{cell(d['recall'],p[1]):>15}  {cell(d['f1'],p[2]):>15}  "
              f"{cell(auc,p[3]):>15}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/mured/ctran_densenet161_poly'
    npz = np.load(os.path.join(run_dir, 'best_preds.npz'), allow_pickle=True)
    label_cols = list(npz['label_cols'])

    val = compute_mured_metrics(npz['val_true'], npz['val_pred'], label_cols)
    test = compute_mured_metrics(npz['test_true'], npz['test_pred'], label_cols)

    print("=" * 78)
    print(f"C-Tran / DenseNet161 on MuReD — report for {run_dir}")
    print("=" * 78)
    headline_table(val, 'VAL')
    headline_table(test, 'TEST')
    print("\nPer-class breakdown (TEST split) vs paper Table XII:")
    per_class_table(test)
    print("\nPaper abstract claim: disease-detection AUC (Bin_AUC) and "
          "disease-classification AUC (ML_AUC).")
    print(f"  Ours  TEST: Bin_AUC={test['Bin_AUC']:.3f}  ML_AUC={test['ML_AUC']:.3f}")
    print(f"  Paper:      Bin_AUC=0.976        ML_AUC=0.962")


if __name__ == '__main__':
    main()
