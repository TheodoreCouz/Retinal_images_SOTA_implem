"""
Re-evaluate trained C-Tran MuReD checkpoints under the RetExpert protocol
(see ./metrics.txt) and write ./performance/<model_name>_mured.md.

Each checkpoint is reloaded and a fresh inference pass is run over the MuReD
TEST split (no augmentation, all-unknown LMT mask = "no prior knowledge").

Usage:
    python eval_performance_mured.py                 # all runs in results/mured
    python eval_performance_mured.py <run_dir> ...   # specific runs
"""
import glob
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import CTranModel
from dataloaders.mured_dataset import MuredDataset
from train_mured import build_transforms
from utils.retexpert_metrics import compute_metrics, recall_at_k

DATA_ROOT = '/storage2/cousin/datasets/MURED'
OUT_DIR = 'performance'


@torch.no_grad()
def predict(model, loader, device='cuda'):
    model.eval()
    probs, targs = [], []
    for batch in loader:
        images = batch['image'].float().to(device)
        mask = batch['mask'].float().to(device)  # all -1 => no prior knowledge
        logits, _, _ = model(images, mask)
        logits = logits.view(batch['labels'].size(0), -1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targs.append(batch['labels'].numpy())
    return np.concatenate(probs), np.concatenate(targs)


def evaluate_run(run_dir, test_df, img_dir, label_cols):
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['args']

    model = CTranModel(num_labels=len(label_cols), use_lmt=a['use_lmt'],
                       pos_emb=False, layers=a['layers'], heads=a['heads'],
                       dropout=a['dropout'], no_x_features=False,
                       backbone=a['backbone']).cuda()
    model.load_state_dict(ck['state_dict'])

    _, eval_tf = build_transforms(a['img_size'])
    ds = MuredDataset(test_df, img_dir, eval_tf, len(label_cols),
                      known_labels=0, testing=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8,
                        pin_memory=True)

    probs, targets = predict(model, loader)
    m = compute_metrics(probs, targets)          # threshold 0.5, average='samples'
    m['recall@5'] = recall_at_k(probs, targets, k=5)
    del model
    torch.cuda.empty_cache()
    return m, ck, a


def write_md(name, m, ck, a, n_test):
    loss_desc = ("PolyLoss (Poly-1, eps={})".format(a['poly_eps'])
                 if a['loss_type'] == 'poly' else "BCE-with-logits")
    eff_bs = a['batch_size'] * a.get('grad_ac_steps', 1)
    bs_desc = str(a['batch_size'])
    if a.get('grad_ac_steps', 1) > 1:
        bs_desc = f"{a['batch_size']} x {a['grad_ac_steps']} grad-accum = {eff_bs} effective"
    sched = a.get('scheduler', 'plateau')
    sched_desc = ("constant (no scheduler — as the paper specifies)" if sched == 'none'
                  else "ReduceLROnPlateau(factor=0.1, patience=5) on val loss")

    lines = [
        f"# C-Tran (DenseNet161) on MuReD — Test Performance",
        "",
        f"Run: `{name}`  •  generated {date.today().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Model: `C-Tran` ({a['layers']} transformer layers, {a['heads']} heads, "
        f"dropout {a['dropout']}, hidden 2208) + `{a['backbone']}` backbone",
        f"- Input: {a['img_size']}x{a['img_size']}, FOV-extracted (black-border crop)",
        f"- Training scheme: LMT (label-mask training) = {a['use_lmt']}; "
        f"inference uses an all-unknown mask (no prior labels)",
        f"- Criterion: {loss_desc}",
        f"- Resampling: LP-ROS {a['lp_ros']}%",
        f"- Optimizer: Adam, LR {a['lr']:.0e}, LR schedule: {sched_desc}",
        f"- Epochs: {a['epochs']}, Batch size: {bs_desc}, Seed: {a['seed']}",
        f"- Best checkpoint selected at epoch: {ck['epoch']} "
        f"(by validation **Model_Score** — see Caveats)",
        f"- Test split: `test_labels_stratified.csv` ({n_test} images, 20 classes)",
        "",
        "## Test set metrics",
        "",
        "Protocol per `metrics.txt` (RetExpert): sigmoid per class, fixed 0.5 "
        "threshold, per-sample averaging for P/R/F1, per-class macro for "
        "AUROC/mAP, flattened Kappa, rank-based Recall@5. All 20 classes "
        "(NORMAL included).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| F1 | {m['f1']:.4f} |",
        f"| mAP | {m['aupr']:.4f} |",
        f"| AUROC | {m['auc']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        f"| Kappa | {m['kappa']:.4f} |",
        f"| Recall@5 | {m['recall@5']:.4f} |",
        "",
        "Additional (in `metrics.txt` but outside the 7-metric summary set):",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy (subset / exact-match) | {m['acc']:.4f} |",
        f"| Hamming loss | {m['hamming']:.4f} |",
        "",
        "## Caveats for cross-model comparison",
        "",
        "- **Checkpoint selection differs.** RetExpert selects the best epoch by "
        "validation F1 (per-sample, 0.5). These C-Tran checkpoints were selected "
        "by validation `Model_Score` (the MuReD paper's own metric: "
        "`((ML_mAP+ML_AUC)/2 + Bin_AUC)/2`), since that was the reproduction "
        "target. Metrics below are otherwise computed under the RetExpert protocol.",
        "- **Precision/Recall/F1 use per-sample averaging**, so they are not "
        "comparable to the `ML_F1` reported in the MuReD paper (which is a "
        "per-class macro over the 19 non-NORMAL classes).",
        "- Inference here runs in **fp32**; RetExpert's `evaluate()` runs under "
        "`autocast` (fp16). `metrics.txt` notes this is a minor, non-bit-exact "
        "difference.",
        "",
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_mured.md")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    return path


def main():
    img_dir = os.path.join(DATA_ROOT, 'images', 'images')
    test_df = pd.read_csv(os.path.join(DATA_ROOT, 'test_labels_stratified.csv'))
    label_cols = list(test_df.columns[1:])

    runs = sys.argv[1:] or sorted(
        os.path.dirname(p) for p in glob.glob('results/mured/*/best_model.pt'))

    rows = []
    for run_dir in runs:
        name = os.path.basename(run_dir)
        m, ck, a = evaluate_run(run_dir, test_df, img_dir, label_cols)
        path = write_md(name, m, ck, a, len(test_df))
        print(f"wrote {path}")
        rows.append((name, m))

    keys = [('f1', 'F1'), ('aupr', 'mAP'), ('auc', 'AUROC'),
            ('precision', 'Precision'), ('recall', 'Recall'),
            ('kappa', 'Kappa'), ('recall@5', 'Recall@5')]
    print("\n" + "=" * 104)
    print(f"{'model':<28}" + "".join(f"{lbl:>11}" for _, lbl in keys))
    print("-" * 104)
    for name, m in rows:
        print(f"{name:<28}" + "".join(f"{m[k]:>11.4f}" for k, _ in keys))
    print("-" * 104)
    print(f"{'RetExpert MuReD (200ep mean)':<28}"
          f"{0.6960:>11.4f}{0.6378:>11.4f}{0.9302:>11.4f}{0.6766:>11.4f}"
          f"{0.7673:>11.4f}{0.6509:>11.4f}{0.9313:>11.4f}")
    print("=" * 104)


if __name__ == '__main__':
    main()
