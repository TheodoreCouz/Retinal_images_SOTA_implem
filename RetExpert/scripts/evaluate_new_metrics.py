"""
Re-evaluates every saved checkpoint under the protocol defined in metrics.txt:
  A. macro F1, macro sensitivity, macro specificity -- per-class thresholds
     t_c tuned on VALIDATION (F1-optimal per class), applied unchanged to TEST.
  B. macro AUROC, macro mAP -- threshold-free, per-class, skip degenerate classes.
  C. Recall@n (adaptive n = n_i per image) and TNR@(m-n)-{1,2,3} -- rank-based,
     threshold-free, per image.

Usage: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python3 scripts/evaluate_new_metrics.py
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_adapter import build_retexpert_vit_large
from utils.datasets import build_dataset
from config import get_dataset_config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Flag-exclusive standard: these are screening/meta flags, not diseases/signs, and are
# dropped by NAME (via config.py's class_names) before any of the 9 metrics are computed.
# MuReD has no such flag and is intentionally absent from this mapping -> left untouched.
FLAG_CLASS = {'RFMiD': 'Disease_Risk', 'PRISM': 'NORMAL'}


def drop_flag_column(dataset_name, probs, targets, class_names):
    """Remove the named meta-flag column from probs/targets (and its entry from
    class_names) if this dataset has one. No-op for datasets not in FLAG_CLASS."""
    flag = FLAG_CLASS.get(dataset_name)
    if flag is None:
        return probs, targets, class_names
    idx = class_names.index(flag)
    keep = [i for i in range(len(class_names)) if i != idx]
    return probs[:, keep], targets[:, keep], [class_names[i] for i in keep]


def get_probs_targets(model, dataset_name, data_path, mode):
    class Args:
        pass
    args = Args()
    args.dataset = dataset_name
    args.data_path = data_path
    args.input_size = 224
    args.color_jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1

    ds = build_dataset(mode, args)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, targets, _ in dl:
            imgs = imgs.to(device)
            out, _ = model(imgs)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def tune_thresholds(val_probs, val_targets):
    """Per-class F1-optimal threshold on validation. Fallback 0.5 if class has no positives."""
    C = val_targets.shape[1]
    thresholds = np.zeros(C)
    for c in range(C):
        y = val_targets[:, c]
        p = val_probs[:, c]
        if y.sum() == 0:
            thresholds[c] = 0.5
            continue
        uniq = np.unique(p)
        if len(uniq) == 1:
            candidates = np.array([0.0, 1.0])
        else:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            candidates = np.concatenate([[0.0], mids, [1.0]])
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            pred = (p >= t).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def macro_threshold_metrics(test_probs, test_targets, thresholds):
    C = test_targets.shape[1]
    preds = (test_probs >= thresholds[None, :]).astype(int)
    f1s, sens, specs = [], [], []
    for c in range(C):
        y, p = test_targets[:, c], preds[:, c]
        tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0))
        fn = np.sum((p == 0) & (y == 1)); tn = np.sum((p == 0) & (y == 0))
        f1s.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0)
        sens.append((tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return float(np.mean(f1s)), float(np.mean(sens)), float(np.mean(specs))


def macro_threshold_free(test_probs, test_targets):
    C = test_targets.shape[1]
    aucs, aps = [], []
    for c in range(C):
        y = test_targets[:, c]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, test_probs[:, c]))
        aps.append(average_precision_score(y, test_probs[:, c]))
    auc = float(np.mean(aucs)) if aucs else 0.0
    ap = float(np.mean(aps)) if aps else 0.0
    return auc, ap


def recall_at_n(test_probs, test_targets):
    N = test_probs.shape[0]
    order = np.argsort(-test_probs, axis=1, kind='stable')  # descending, ties -> ascending class idx
    scores = []
    for i in range(N):
        n_i = int(test_targets[i].sum())
        if n_i == 0:
            continue
        top_i = set(order[i, :n_i].tolist())
        pos_i = set(np.where(test_targets[i] == 1)[0].tolist())
        scores.append(len(top_i & pos_i) / n_i)
    return float(np.mean(scores)) if scores else 0.0


def tnr_at_mn_j(test_probs, test_targets, j):
    N, C = test_probs.shape
    order = np.argsort(-test_probs, axis=1, kind='stable')
    scores = []
    for i in range(N):
        n_i = int(test_targets[i].sum())
        neg_i = C - n_i
        k = neg_i - j
        if k < 1:
            continue
        bot_i = set(order[i, -k:].tolist())
        neg_set = set(np.where(test_targets[i] == 0)[0].tolist())
        scores.append(len(bot_i & neg_set) / k)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_run(dataset_name, nb_classes, data_path, ckpt_path):
    model = build_retexpert_vit_large(num_classes=nb_classes, adapter_mode='AKU', adapter_dim=64)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    val_probs, val_targets = get_probs_targets(model, dataset_name, data_path, 'val')
    test_probs, test_targets = get_probs_targets(model, dataset_name, data_path, 'test')

    class_names = get_dataset_config(dataset_name)['class_names']
    val_probs, val_targets, class_names_check = drop_flag_column(dataset_name, val_probs, val_targets, class_names)
    test_probs, test_targets, class_names = drop_flag_column(dataset_name, test_probs, test_targets, class_names)
    assert class_names_check == class_names  # val/test must drop the identical column

    thresholds = tune_thresholds(val_probs, val_targets)
    macro_f1, macro_sens, macro_spec = macro_threshold_metrics(test_probs, test_targets, thresholds)
    macro_auroc, macro_map = macro_threshold_free(test_probs, test_targets)
    r_at_n = recall_at_n(test_probs, test_targets)
    tnr1 = tnr_at_mn_j(test_probs, test_targets, 1)
    tnr2 = tnr_at_mn_j(test_probs, test_targets, 2)
    tnr3 = tnr_at_mn_j(test_probs, test_targets, 3)

    return {
        'macro_f1': macro_f1, 'macro_sensitivity': macro_sens, 'macro_specificity': macro_spec,
        'macro_auroc': macro_auroc, 'macro_map': macro_map,
        'recall_at_n': r_at_n, 'tnr_at_mn_1': tnr1, 'tnr_at_mn_2': tnr2, 'tnr_at_mn_3': tnr3,
        'thresholds': {cn: float(t) for cn, t in zip(class_names, thresholds)},
    }


def build_runs(datasets=None):
    # 200 epochs only. A dedicated experiment confirmed training longer (400 epochs)
    # does not improve performance on any of the three datasets; those checkpoints
    # were deleted afterward and are intentionally absent here.
    runs = []
    for s in [42, 43, 44, 45, 46]:
        runs.append(dict(dataset='MuReD', nb_classes=20, data_path='./data/MuReD', epochs=200, seed=s,
                          ckpt=f'./output_dir/mured_retexpert_seed{s}/checkpoint-best.pth'))
    runs.append(dict(dataset='RFMiD', nb_classes=29, data_path='./data/RFMiD', epochs=200, seed=42,
                      ckpt='./output_dir/rfmid_retexpert_run2/checkpoint-best.pth'))
    for s in [43, 44, 45, 46]:
        runs.append(dict(dataset='RFMiD', nb_classes=29, data_path='./data/RFMiD', epochs=200, seed=s,
                          ckpt=f'./output_dir/rfmid_retexpert_seed{s}_200ep/checkpoint-best.pth'))
    for s in [42, 43, 44, 45, 46]:
        runs.append(dict(dataset='PRISM', nb_classes=32, data_path='./data/PRISM', epochs=200, seed=s,
                          ckpt=f'./output_dir/prism_retexpert_seed{s}_200ep/checkpoint-best.pth'))
    if datasets is not None:
        runs = [r for r in runs if r['dataset'] in datasets]
    return runs


COMBINED_JSON = './output_dir/new_metrics_combined_results.json'

if __name__ == '__main__':
    # Flag-exclusive re-evaluation of RFMiD/PRISM only. MuReD has no meta-flag class
    # and is left completely untouched (not re-run, not overwritten).
    TARGET_DATASETS = ['RFMiD', 'PRISM']

    old_all = json.load(open(COMBINED_JSON)) if os.path.exists(COMBINED_JSON) else []
    old_by_key = {(r['dataset'], r['epochs'], r['seed']): r for r in old_all}
    kept_untouched = [r for r in old_all if r['dataset'] not in TARGET_DATASETS]

    new_results = []
    for run in build_runs(datasets=TARGET_DATASETS):
        print(f"Evaluating {run['dataset']} seed={run['seed']} epochs={run['epochs']} "
              f"(flag column dropped: {FLAG_CLASS.get(run['dataset'])}) ...", flush=True)
        r = evaluate_run(run['dataset'], run['nb_classes'], run['data_path'], run['ckpt'])
        row = dict(dataset=run['dataset'], epochs=run['epochs'], seed=run['seed'], **r)
        new_results.append(row)
        print("  -> macro_f1={macro_f1:.4f} sens={macro_sensitivity:.4f} spec={macro_specificity:.4f} "
              "auroc={macro_auroc:.4f} map={macro_map:.4f} recall@n={recall_at_n:.4f} "
              "tnr1={tnr_at_mn_1:.4f} tnr2={tnr_at_mn_2:.4f} tnr3={tnr_at_mn_3:.4f}".format(**r))
        # Save incrementally: untouched MuReD entries + whatever RFMiD/PRISM results exist so far.
        with open(COMBINED_JSON, 'w') as f:
            json.dump(kept_untouched + new_results, f, indent=2)

    # Sanity-check report: before (old, flag-included) vs after (new, flag-excluded).
    print("\n" + "=" * 70)
    print("BEFORE/AFTER sanity check (flag column dropped)")
    print("=" * 70)
    metrics_order = ['macro_f1', 'macro_sensitivity', 'macro_specificity', 'macro_auroc',
                      'macro_map', 'recall_at_n', 'tnr_at_mn_1', 'tnr_at_mn_2', 'tnr_at_mn_3']
    for row in new_results:
        key = (row['dataset'], row['epochs'], row['seed'])
        old = old_by_key.get(key)
        print(f"\n{row['dataset']} seed={row['seed']} epochs={row['epochs']}:")
        if old is None:
            print("  (no prior 'before' entry found)")
            continue
        for m in metrics_order:
            print(f"  {m:20s} before={old[m]:.4f}  after={row[m]:.4f}  delta={row[m]-old[m]:+.4f}")

    print(f"\nSaved combined results to {COMBINED_JSON} (MuReD entries left untouched)")
