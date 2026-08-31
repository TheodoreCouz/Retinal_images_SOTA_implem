import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from types import SimpleNamespace

from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet

PRISM_DIR = "/home/cousin/research/Fiber_dino/PRISM v1"


@torch.no_grad()
def predict(model, loader):
    preds = []
    for x, _ in loader:
        x = x.cuda()
        with torch.cuda.amp.autocast():
            out = model(x)
        preds.append(out.sigmoid().float().cpu().numpy())
    return np.concatenate(preds)


def evaluate(ckpt_path, split_name, img_dir, label_csv, num_classes=31):
    df = pd.read_csv(label_csv)
    label_cols = list(df.columns[1:])
    args = SimpleNamespace(img_size=960)

    model = create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=num_classes)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.cuda().eval()

    ds = PrismDataSet(image_ids=df, baseImgPath=img_dir)
    ds.transform = get_riadd_valid_transforms(args)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)

    y_true = df[label_cols].values.astype(np.int64)
    y_pred = predict(model, loader)

    per_class = {}
    aucs, maps = [], []
    for i, col in enumerate(label_cols):
        yt, yp = y_true[:, i], y_pred[:, i]
        if len(np.unique(yt)) < 2:
            per_class[col] = {"auc": None, "ap": None, "n_pos": int(yt.sum())}
            continue
        auc = roc_auc_score(yt, yp)
        ap = average_precision_score(yt, yp)
        aucs.append(auc)
        maps.append(ap)
        per_class[col] = {"auc": round(float(auc), 4), "ap": round(float(ap), 4), "n_pos": int(yt.sum())}

    mean_auc = float(np.mean(aucs))
    mean_ap = float(np.mean(maps))
    print(f"\n=== {split_name} (best@epoch {ck['epoch']}, ckpt training-time score {ck['metric']:.4f}) ===")
    print(f"n_images={len(df)}  n_classes_scored={len(aucs)}/{len(label_cols)}")
    print(f"mean AUC = {mean_auc:.4f}   mAP = {mean_ap:.4f}")
    for col, m in per_class.items():
        if m["auc"] is not None:
            print(f"  {col:6s} n_pos={m['n_pos']:>3d}  AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
        else:
            print(f"  {col:6s} n_pos={m['n_pos']:>3d}  (skipped: single-class)")

    return dict(mean_auc=mean_auc, mAP=mean_ap, per_class=per_class,
                best_epoch=ck['epoch'], training_score=float(ck['metric']))


if __name__ == '__main__':
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt_path is None:
        print("Usage: python eval_prism.py <path/to/model_best.pth.tar>")
        sys.exit(1)

    val_res = evaluate(ckpt_path, 'PRISM Validation', f'{PRISM_DIR}/val', f'{PRISM_DIR}/val_labels.csv')
    test_res = evaluate(ckpt_path, 'PRISM Test', f'{PRISM_DIR}/test', f'{PRISM_DIR}/test_labels.csv')
