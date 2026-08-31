"""
One-off script: reload each saved checkpoint, run inference on its test set,
and compute Recall@5 (per-sample: fraction of true-positive labels captured
within the top-5 highest-probability predicted classes, averaged over samples
that have at least one positive label). Combines this with the F1/Precision/
Recall/AUROC/mAP/Kappa already recorded in each run's performance JSON.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_adapter import build_retexpert_vit_large
from utils.datasets import build_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def evaluate_checkpoint(dataset_name, nb_classes, data_path, ckpt_path):
    model = build_retexpert_vit_large(num_classes=nb_classes, adapter_mode='AKU', adapter_dim=64)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

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

    ds_test = build_dataset('test', args)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=8)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, targets, _ in dl_test:
            imgs = imgs.to(device)
            out, _ = model(imgs)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r5 = recall_at_k(all_probs, all_targets, k=5)
    return r5, ckpt.get('epoch')


RUNS = []
# MuReD 200ep
for s in [42, 43, 44, 45, 46]:
    RUNS.append(dict(
        dataset='MuReD', nb_classes=20, data_path='./data/MuReD', epochs=200, seed=s,
        ckpt=f'./output_dir/mured_retexpert_seed{s}/checkpoint-best.pth',
        perf_json=f'./performance/MuReD_retexpert_seed{s}_test_performance.json' if s != 42
                   else './performance/MuReD_retexpert_seed42_200ep_test_performance.json',
    ))
# MuReD 400ep
for s in [42, 43, 44, 45, 46]:
    RUNS.append(dict(
        dataset='MuReD', nb_classes=20, data_path='./data/MuReD', epochs=400, seed=s,
        ckpt=f'./output_dir/mured_retexpert_seed{s}_400ep_full/checkpoint-best.pth',
        perf_json=f'./performance/MuReD_retexpert_seed{s}_ep400_test_performance.json',
    ))
# RFMiD 200ep (single seed)
RUNS.append(dict(
    dataset='RFMiD', nb_classes=29, data_path='./data/RFMiD', epochs=200, seed=42,
    ckpt='./output_dir/rfmid_retexpert_run2/checkpoint-best.pth',
    perf_json='./performance/RFMiD_retexpert_test_performance.json',
))
# RFMiD 400ep
for s in [42, 43, 44, 45, 46]:
    RUNS.append(dict(
        dataset='RFMiD', nb_classes=29, data_path='./data/RFMiD', epochs=400, seed=s,
        ckpt=f'./output_dir/rfmid_retexpert_seed{s}_400ep/checkpoint-best.pth',
        perf_json=f'./performance/RFMiD_retexpert_seed{s}_ep400_test_performance.json',
    ))

results = []
for run in RUNS:
    print(f"Evaluating {run['dataset']} seed={run['seed']} epochs={run['epochs']} ...")
    r5, ckpt_epoch = evaluate_checkpoint(run['dataset'], run['nb_classes'], run['data_path'], run['ckpt'])
    perf = json.load(open(run['perf_json']))['test_metrics']
    row = dict(
        dataset=run['dataset'], epochs=run['epochs'], seed=run['seed'],
        f1=perf['f1'], mAP=perf['aupr'], auroc=perf['auc'],
        precision=perf['precision'], recall=perf['recall'], kappa=perf['kappa'],
        recall_at_5=r5,
    )
    results.append(row)
    print("  ->", row)

with open('./output_dir/recall_at_5_combined_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved combined results to ./output_dir/recall_at_5_combined_results.json")
