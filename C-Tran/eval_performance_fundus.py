"""
Re-inference evaluator for a trained C-Tran fundus checkpoint under the Fiber
9-metric protocol (metrics.txt), with the FLAG-EXCLUSIVE class-set standard.

Reloads the checkpoint, rebuilds the exact (val, test) splits, runs fresh
inference (all-unknown LMT mask = no prior knowledge) to obtain full-class
prob/target matrices, then DROPS the screening-flag column BY NAME from both
probs and targets before calling compute_fiber_metrics — so threshold tuning
and all 9 metrics run on the reduced set:
    MuReD 20 (unchanged) | RFMiD 28 (drop Disease_Risk) | PRISM 31 (drop NORMAL).

For C-Tran the cached full-class probs in best_preds.npz make re-inference
redundant (dropping the column from cached probs is identical); this script
exists for models without cached probs and to VERIFY the cached numbers.

Usage: python eval_performance_fundus.py results/<dataset>/<name> [...] [--verify]
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CTranModel
from dataloaders.mured_dataset import MuredDataset
from dataloaders.fundus_prep import PREP
from train_mured import build_transforms
from utils.fiber_metrics import compute_fiber_metrics, METRIC_KEYS

DROP_FLAG = {'mured': None, 'rfmid': 'Disease_Risk', 'prism': 'NORMAL'}


@torch.no_grad()
def _infer(model, df, num_labels, img_size, device='cuda'):
    _, eval_tf = build_transforms(img_size)
    ds = MuredDataset(df, '', eval_tf, num_labels, known_labels=0, testing=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8,
                        pin_memory=True)
    model.eval()
    probs, targs = [], []
    for batch in loader:
        images = batch['image'].float().to(device)
        mask = batch['mask'].float().to(device)
        logits, _, _ = model(images, mask)
        logits = logits.view(batch['labels'].size(0), -1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targs.append(batch['labels'].numpy())
    return np.concatenate(probs), np.concatenate(targs)


def drop_flag(probs, targets, label_cols, flag):
    """Drop the named flag column from probs AND targets (returns reduced set)."""
    if flag is None:
        return probs, targets, label_cols
    assert flag in label_cols, f"flag '{flag}' not in {label_cols}"
    fi = label_cols.index(flag)
    keep = [i for i in range(len(label_cols)) if i != fi]
    return probs[:, keep], targets[:, keep], [label_cols[i] for i in keep]


def evaluate_run(run_dir):
    """Re-infer val+test, drop the dataset's flag column, return Fiber metrics."""
    dataset = os.path.basename(os.path.dirname(run_dir))
    flag = DROP_FLAG.get(dataset)
    ck = torch.load(os.path.join(run_dir, 'best_model.pt'),
                    map_location='cpu', weights_only=False)
    a = ck['args']
    label_cols = [str(c) for c in ck['label_cols']]

    _, val_df, test_df, prep_cols = PREP[dataset]()   # deterministic splits
    assert [str(c) for c in prep_cols] == label_cols, "label-col mismatch"

    model = CTranModel(num_labels=len(label_cols), use_lmt=a['use_lmt'],
                       pos_emb=False, layers=a['layers'], heads=a['heads'],
                       dropout=a['dropout'], no_x_features=False,
                       backbone=a['backbone']).cuda()
    model.load_state_dict(ck['state_dict'])

    vp, vt = _infer(model, val_df, len(label_cols), a['img_size'])
    tp, tt = _infer(model, test_df, len(label_cols), a['img_size'])
    del model
    torch.cuda.empty_cache()

    # drop the flag column BY NAME from BOTH probs and targets, then evaluate
    vp, vt, _ = drop_flag(vp, vt, label_cols, flag)
    tp, tt, red_cols = drop_flag(tp, tt, label_cols, flag)
    m = compute_fiber_metrics(vt, vp, tt, tp)
    m['_n_classes'] = len(red_cols)
    m['_flag'] = flag
    return m


def cached_reduced(run_dir):
    """Same metrics but from cached best_preds.npz (for verification)."""
    dataset = os.path.basename(os.path.dirname(run_dir))
    flag = DROP_FLAG.get(dataset)
    z = np.load(os.path.join(run_dir, 'best_preds.npz'), allow_pickle=True)
    lc = [str(c) for c in z['label_cols']]
    vp, vt, _ = drop_flag(z['val_pred'], z['val_true'], lc, flag)
    tp, tt, _ = drop_flag(z['test_pred'], z['test_true'], lc, flag)
    return compute_fiber_metrics(vt, vp, tt, tp)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    verify = '--verify' in sys.argv
    for run_dir in args:
        run_dir = run_dir.rstrip('/')
        m = evaluate_run(run_dir)
        print(f"\n{run_dir}  ({m['_n_classes']} classes, dropped {m['_flag']})")
        for key, lbl, _ in METRIC_KEYS:
            print(f"  {lbl:16s} {m[key]:.4f}")
        if verify:
            c = cached_reduced(run_dir)
            dr = abs(m['Recall@n'] - c['Recall@n'])
            df1 = abs(m['macro_F1'] - c['macro_F1'])
            print(f"  [verify vs cached]  Recall@n reinfer={m['Recall@n']:.4f} "
                  f"cached={c['Recall@n']:.4f} |Δ|={dr:.2e}  ;  "
                  f"macroF1 |Δ|={df1:.2e}")


if __name__ == '__main__':
    main()
