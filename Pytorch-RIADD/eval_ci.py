"""Evaluate the CI ensemble job: per-seed 5-fold ensembles -> 9 metrics -> 95% CI.

For each dataset and seed, average the 5 folds' sigmoid outputs on val+test, tune the
per-class F1-optimal thresholds on val, apply to test, and compute the 9-metric Fiber
protocol (metrics9 / metrics.txt). Then aggregate the per-seed test metrics into
mean +/- t-based 95% CI. PRISM has a single ensemble (seed 42) -> point estimate, no CI.

Caches per-fold probs to performance/preds_ci/*.npy so re-runs need no inference.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from scipy import stats

from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet, RiaddDataSet
from metrics9 import compute_all, METRIC_KEYS

PRED_DIR = "performance/preds_ci"
BATCH_SIZE = 8
SEEDS = [42, 43, 44, 45, 46]

MURED_DIR = "/storage2/cousin/datasets/MURED"
PRISM_DIR = "/home/cousin/research/Fiber_dino/PRISM v1"
RFMID_DIR = "/storage2/cousin/datasets/RFMiD"

# dataset -> (ckpt prefix, num_classes, dataset_class, seeds, disease_start_idx,
#             (val_imgdir, val_csv), (test_imgdir, test_csv))
DATASETS = {
    "mured": dict(
        prefix="ckpt/tf_efficientnet_b6_ns-MURED-KFOLD-SGD1E-1-seed", nc=20, ds=PrismDataSet,
        seeds=SEEDS, disease_start=0,
        val=(f"{MURED_DIR}/images/images", f"{MURED_DIR}/val_labels.csv"),
        test=(f"{MURED_DIR}/images/images", f"{MURED_DIR}/test_labels.csv")),
    "rfmid": dict(
        prefix="ckpt/tf_efficientnet_b6_ns-RIADD-KFOLD-SGD1E-1-seed", nc=29, ds=RiaddDataSet,
        seeds=SEEDS, disease_start=1,  # drop index-0 Disease_Risk for the reported metrics
        val=(f"{RFMID_DIR}/Validation", f"{RFMID_DIR}/validation_labels_29.csv"),
        test=(f"{RFMID_DIR}/Test", f"{RFMID_DIR}/testing_labels_29.csv")),
    "prism": dict(
        prefix="ckpt/tf_efficientnet_b6_ns-PRISM-KFOLD-SGD1E-1-seed", nc=31, ds=PrismDataSet,
        seeds=[42], disease_start=0,
        val=(f"{PRISM_DIR}/val", f"{PRISM_DIR}/val_labels.csv"),
        test=(f"{PRISM_DIR}/test", f"{PRISM_DIR}/test_labels.csv")),
}


def fold_ckpts(prefix, seed):
    paths = []
    for f in range(5):
        hits = sorted(glob.glob(f"{prefix}{seed}fold_{f}/train/*/model_best.pth.tar"))
        if not hits:
            return None
        paths.append(hits[-1])  # latest timestamp if a fold was retrained
    return paths


def load_model(ckpt_path, nc):
    m = create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=nc)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ck['state_dict'].items()}
    m.load_state_dict(sd, strict=True)
    m.cuda().eval()
    return m


@torch.no_grad()
def predict(model, loader):
    out = []
    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            out.append(model(x).sigmoid().float().cpu().numpy())
    return np.concatenate(out)


def ensemble_probs(cfg, seed, split):
    """Mean-sigmoid over the 5 folds for one seed/split. Cached per (seed, split)."""
    cache = f"{PRED_DIR}/{cfg['prefix'].split('/')[-1]}{seed}_{split}.npy"
    img_dir, csv = cfg[split]
    df = pd.read_csv(csv)
    y = df[df.columns[1:]].values.astype(np.int64)
    if os.path.exists(cache):
        return np.load(cache), y, list(df.columns[1:])

    ckpts = fold_ckpts(cfg["prefix"], seed)
    if ckpts is None:
        return None, y, list(df.columns[1:])
    ds = cfg["ds"](image_ids=df, baseImgPath=img_dir)
    ds.transform = get_riadd_valid_transforms(SimpleNamespace(img_size=960))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    fold_p = []
    for cp in ckpts:
        m = load_model(cp, cfg["nc"])
        fold_p.append(predict(m, loader))
        del m
        torch.cuda.empty_cache()
    ens = np.mean(fold_p, axis=0)
    np.save(cache, ens)
    return ens, y, list(df.columns[1:])


def eval_seed(cfg, seed):
    d0 = cfg["disease_start"]
    val_p, val_y, cols = ensemble_probs(cfg, seed, "val")
    test_p, test_y, _ = ensemble_probs(cfg, seed, "test")
    if val_p is None or test_p is None:
        return None
    m = compute_all(val_p[:, d0:], val_y[:, d0:], test_p[:, d0:], test_y[:, d0:])
    return dict(seed=seed, metrics=m)


def ci(values):
    a = np.asarray(values, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return dict(mean=mean, std=None, n=n, ci95_lo=None, ci95_hi=None, half_width=None)
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(n)
    h = float(stats.t.ppf(0.975, n - 1) * se)
    return dict(mean=mean, std=sd, n=n, ci95_lo=mean - h, ci95_hi=mean + h, half_width=h)


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    results = {}
    for name, cfg in DATASETS.items():
        print(f"=== {name} ===")
        per_seed = []
        for s in cfg["seeds"]:
            r = eval_seed(cfg, s)
            if r is None:
                print(f"  seed {s}: checkpoints not ready, skipping")
                continue
            per_seed.append(r)
            tm = r["metrics"]
            print(f"  seed {s}: " + "  ".join(f"{k}={tm[k]:.4f}" for k in METRIC_KEYS))
        if not per_seed:
            results[name] = {"status": "no checkpoints yet"}
            continue
        agg = {k: ci([r["metrics"][k] for r in per_seed]) for k in METRIC_KEYS}
        results[name] = {"n_ensembles": len(per_seed), "per_seed": per_seed, "test_ci": agg}
        print(f"  --> {name} test CI (n={len(per_seed)}):")
        for k in METRIC_KEYS:
            c = agg[k]
            if c["ci95_lo"] is None:
                print(f"       {k:18s} {c['mean']:.4f} (single run, no CI)")
            else:
                print(f"       {k:18s} {c['mean']:.4f}  95% CI [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]  (±{c['half_width']:.4f})")

    with open("performance/ci_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote performance/ci_metrics.json")


if __name__ == "__main__":
    sys.exit(main())
