"""Compute the 7-metric set (metrics.txt protocol) for MURED, RFMiD and PRISM.

Caches raw sigmoid probabilities to performance/preds/*.npy so metrics can be
recomputed without re-running inference.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet, RiaddDataSet
from metrics7 import compute_metrics, per_class_table, find_best_threshold

PRED_DIR = "performance/preds"
BATCH_SIZE = 4

MURED_DIR = "/storage2/cousin/datasets/MURED"
PRISM_DIR = "/home/cousin/research/Fiber_dino/PRISM v1"
RFMID_DIR = "/storage2/cousin/datasets/RFMiD"

MURED_CKPT = "ckpt/tf_efficientnet_b6_ns-MURED-SGD1E-1/train/20260713-165204-tf_efficientnet_b6_ns-960/model_best.pth.tar"
PRISM_CKPT = "ckpt/tf_efficientnet_b6_ns-PRISM-SGD1E-1/train/20260703-013634-tf_efficientnet_b6_ns-960/model_best.pth.tar"
RFMID_FOLD_CKPTS = {
    0: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_0/train/20260701-140832-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    1: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_1/train/20260702-114241-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    2: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_2/train/20260702-121737-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    3: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_3/train/20260702-125224-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    4: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_4/train/20260702-132710-tf_efficientnet_b6_ns-960/model_best.pth.tar",
}


def load_model(ckpt_path, num_classes):
    model = create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=num_classes)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.cuda().eval()
    return model, ck['epoch'], float(ck['metric'])


@torch.no_grad()
def predict(model, loader):
    preds = []
    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(x)
        preds.append(out.sigmoid().float().cpu().numpy())
    return np.concatenate(preds)


def make_loader(ds_cls, df, img_dir):
    ds = ds_cls(image_ids=df, baseImgPath=img_dir)
    ds.transform = get_riadd_valid_transforms(SimpleNamespace(img_size=960))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)


def cached_predict(tag, fn):
    path = f"{PRED_DIR}/{tag}.npy"
    if os.path.exists(path):
        print(f"  [cache] {tag}")
        return np.load(path)
    probs = fn()
    np.save(path, probs)
    return probs


def single_probs(name, ckpt, ds_cls, img_dir, label_csv, split, num_classes):
    df = pd.read_csv(label_csv)
    label_cols = list(df.columns[1:])
    y_true = df[label_cols].values.astype(np.int64)

    def infer():
        model, epoch, metric = load_model(ckpt, num_classes)
        print(f"  {name}/{split}: ckpt best@epoch {epoch} (training score {metric:.4f})")
        p = predict(model, make_loader(ds_cls, df, img_dir))
        del model
        torch.cuda.empty_cache()
        return p

    return cached_predict(f"{name}_{split}", infer), y_true, label_cols


def rfmid_ensemble_probs(split, img_dir, label_csv):
    df = pd.read_csv(label_csv)
    label_cols = list(df.columns[1:])
    y_true = df[label_cols].values.astype(np.int64)

    fold_probs = []
    for fold, ckpt in RFMID_FOLD_CKPTS.items():
        def infer(ckpt=ckpt, fold=fold):
            model, epoch, metric = load_model(ckpt, 29)
            print(f"  rfmid/{split} fold {fold}: ckpt best@epoch {epoch} (internal AUC {metric:.4f})")
            p = predict(model, make_loader(RiaddDataSet, df, img_dir))
            del model
            torch.cuda.empty_cache()
            return p
        fold_probs.append(cached_predict(f"rfmid_{split}_fold{fold}", infer))

    ens = np.mean(fold_probs, axis=0)
    np.save(f"{PRED_DIR}/rfmid_{split}_ensemble.npy", ens)
    return ens, y_true, label_cols


def tuned_dataset(name, val, test):
    """val/test are (probs, y_true, label_cols). Tune F1-optimal threshold on val,
    apply unchanged to test. Returns per-split result dicts + the chosen threshold."""
    val_p, val_y, cols = val
    test_p, test_y, _ = test
    thr, val_f1_at_thr = find_best_threshold(val_p, val_y)
    print(f"  {name}: F1-optimal global threshold (tuned on val) = {thr:.2f} "
          f"(val per-sample F1 {val_f1_at_thr:.4f})")
    out = {"pr_threshold": thr, "classes": cols}
    for split, (p, y) in [("val", (val_p, val_y)), ("test", (test_p, test_y))]:
        out[split] = {
            "metrics": compute_metrics(p, y, pr_threshold=thr),
            "per_class": per_class_table(p, y, cols, threshold=thr),
        }
    return out


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    results = {}

    print("=== MURED ===")
    results["mured"] = tuned_dataset(
        "mured",
        single_probs("mured", MURED_CKPT, PrismDataSet, f"{MURED_DIR}/images/images", f"{MURED_DIR}/val_labels.csv", "val", 20),
        single_probs("mured", MURED_CKPT, PrismDataSet, f"{MURED_DIR}/images/images", f"{MURED_DIR}/test_labels.csv", "test", 20),
    )

    print("=== PRISM ===")
    results["prism"] = tuned_dataset(
        "prism",
        single_probs("prism", PRISM_CKPT, PrismDataSet, f"{PRISM_DIR}/val", f"{PRISM_DIR}/val_labels.csv", "val", 31),
        single_probs("prism", PRISM_CKPT, PrismDataSet, f"{PRISM_DIR}/test", f"{PRISM_DIR}/test_labels.csv", "test", 31),
    )

    print("=== RFMiD (5-fold ensemble) ===")
    val_p, val_y, cols = rfmid_ensemble_probs("val", f"{RFMID_DIR}/Validation", f"{RFMID_DIR}/validation_labels_29.csv")
    test_p, test_y, _ = rfmid_ensemble_probs("test", f"{RFMID_DIR}/Test", f"{RFMID_DIR}/testing_labels_29.csv")
    # Primary: 28 disease classes (index 0 is the Disease_Risk screening flag, not a disease
    # -- excluded so numbers are comparable to MURED/PRISM). Each class set gets its own
    # validation-tuned threshold since the label set (and thus optimal cutoff) differs.
    results["rfmid"] = tuned_dataset(
        "rfmid (28 disease)",
        (val_p[:, 1:], val_y[:, 1:], cols[1:]),
        (test_p[:, 1:], test_y[:, 1:], cols[1:]),
    )
    results["rfmid"]["classes"] = cols  # keep full 29-name list for reference
    rfmid29 = tuned_dataset(
        "rfmid (29 incl. disease-risk)",
        (val_p, val_y, cols),
        (test_p, test_y, cols),
    )
    for split in ("val", "test"):
        results["rfmid"][split]["metrics_incl_disease_risk"] = rfmid29[split]["metrics"]
    results["rfmid"]["pr_threshold_incl_disease_risk"] = rfmid29["pr_threshold"]

    with open("performance/all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    for ds in ["mured", "rfmid", "prism"]:
        m = results[ds]["test"]["metrics"]
        print(f"\n{ds.upper()} test (thr={m['pr_threshold']:.2f}): " + "  ".join(
            f"{k}={m[k]:.4f}" for k in ["f1", "mAP", "auroc", "precision", "recall", "kappa", "recall_at_5"]))


if __name__ == "__main__":
    sys.exit(main())
