import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from types import SimpleNamespace

from timm.models import create_model
from timm.data import get_riadd_valid_transforms, RiaddDataSet

FOLD_CKPTS = {
    0: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_0/train/20260701-140832-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    1: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_1/train/20260702-114241-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    2: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_2/train/20260702-121737-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    3: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_3/train/20260702-125224-tf_efficientnet_b6_ns-960/model_best.pth.tar",
    4: "ckpt/tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V4-SGD1E-1fold_4/train/20260702-132710-tf_efficientnet_b6_ns-960/model_best.pth.tar",
}
BASE = "/storage2/cousin/datasets/RFMiD"
LABEL_COLS = None  # filled in from csv


def load_model(ckpt_path):
    m = create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=29)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ck['state_dict'].items()}
    m.load_state_dict(sd, strict=True)
    m.cuda().eval()
    return m, ck['epoch'], ck['metric']


@torch.no_grad()
def predict(model, loader):
    preds = []
    for x, _ in loader:
        x = x.cuda()
        with torch.cuda.amp.autocast():
            out = model(x)
        preds.append(out.sigmoid().float().cpu().numpy())
    return np.concatenate(preds)


def multi_disease_avg_score(y_true, y_pred, disease_idx):
    aucs, maps = [], []
    for i in disease_idx:
        yt, yp = y_true[:, i], y_pred[:, i]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        maps.append(average_precision_score(yt, yp))
    return 0.5 * (np.mean(aucs) + np.mean(maps)), np.mean(aucs), np.mean(maps), len(aucs)


def evaluate_split(split_name, img_dir, label_csv):
    global LABEL_COLS
    df = pd.read_csv(label_csv)
    LABEL_COLS = list(df.columns[1:])
    args = SimpleNamespace(img_size=960)
    ds = RiaddDataSet(image_ids=df, baseImgPath=img_dir)
    ds.transform = get_riadd_valid_transforms(args)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)

    y_true = df[LABEL_COLS].values.astype(np.int64)

    fold_preds = {}
    print(f"\n=== {split_name} set ({len(df)} images) ===")
    for fold_i, ckpt in FOLD_CKPTS.items():
        model, epoch, metric = load_model(ckpt)
        preds = predict(model, loader)
        fold_preds[fold_i] = preds
        risk_auc = roc_auc_score(y_true[:, 0], preds[:, 0])
        mds, mauc, mmap, n = multi_disease_avg_score(y_true, preds, range(1, 29))
        print(f"  fold {fold_i} (best@epoch {epoch:>2}, internal AUC {metric:.4f}): "
              f"Disease-Risk AUC={risk_auc:.4f}  Multi-Disease Avg={mds:.4f} (AUC={mauc:.4f} mAP={mmap:.4f}, n={n}/28)")
        del model
        torch.cuda.empty_cache()

    ens = np.mean(list(fold_preds.values()), axis=0)
    risk_auc = roc_auc_score(y_true[:, 0], ens[:, 0])
    mds, mauc, mmap, n = multi_disease_avg_score(y_true, ens, range(1, 29))
    final_score = 0.5 * (risk_auc + mds)
    print(f"  ENSEMBLE (mean of 5 folds): Disease-Risk AUC={risk_auc:.4f}  "
          f"Multi-Disease Avg={mds:.4f} (AUC={mauc:.4f} mAP={mmap:.4f})  "
          f"Final Score={final_score:.4f}")
    return dict(risk_auc=risk_auc, mds=mds, mauc=mauc, mmap=mmap, final=final_score)


if __name__ == '__main__':
    val_res = evaluate_split('Validation', f'{BASE}/Validation', f'{BASE}/validation_labels_29.csv')
    test_res = evaluate_split('Testing', f'{BASE}/Test', f'{BASE}/testing_labels_29.csv')
