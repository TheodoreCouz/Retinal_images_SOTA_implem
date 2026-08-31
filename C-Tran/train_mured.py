"""
Reproduction of C-Tran on the MuReD dataset, following
Rodriguez et al., "Multi-Label Retinal Disease Classification Using
Transformers" (IEEE JBHI 2023).

Optimal configuration reported by the paper (Section V.B):
  backbone = DenseNet161, Adam, lr = 1e-5, 3 transformer layers, dropout 0.1,
  image size 384x384, PolyLoss, LP-ROS 10% resampling, LMT training,
  no prior labels at inference.
"""
import argparse
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import CTranModel
from models.utils import custom_replace
from dataloaders.mured_dataset import MuredDataset
from dataloaders.resampling import lp_ros
from utils.mured_metrics import compute_mured_metrics, format_metrics, format_per_class

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentations (paper Table III); FOV crop is done inside the dataset.
# ---------------------------------------------------------------------------
def build_transforms(img_size):
    train_tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.MedianBlur(blur_limit=7, p=0.3),
        A.GaussNoise(std_range=(0.0, 0.15), p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,
                             val_shift_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(num_holes_range=(5, 5),
                        hole_height_range=(20, 20), hole_width_range=(20, 20),
                        fill=0, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Loss functions.  Per-element output; the LMT machinery masks + sums.
# ---------------------------------------------------------------------------
def bce_per_elem(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')


def poly1_per_elem(logits, targets, epsilon=1.0):
    """Poly-1 cross entropy (Leng et al. 2022) on top of BCE."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    return ce + epsilon * (1 - pt)


def compute_loss(logits, targets, unk_mask, loss_type, poly_eps):
    if loss_type == 'poly':
        per = poly1_per_elem(logits, targets, poly_eps)
    else:
        per = bce_per_elem(logits, targets)
    # LMT: only unknown labels contribute to the loss (paper's scheme).
    return (unk_mask * per).sum()


# ---------------------------------------------------------------------------
# One epoch (train or eval).  Returns sigmoid-prob preds, targets, avg loss.
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, loss_type, poly_eps,
              train, grad_ac_steps=1):
    model.train() if train else model.eval()
    if train:
        optimizer.zero_grad()

    all_preds, all_targs = [], []
    loss_total, n_seen = 0.0, 0

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].float().to(device)
        labels = batch['labels'].float().to(device)
        mask = batch['mask'].float().to(device)
        unk_mask = custom_replace(mask, 1, 0, 0)  # 1 at unknown positions

        with torch.set_grad_enabled(train):
            logits, _, _ = model(images, mask.clone())
            logits = logits.view(labels.size(0), -1)
            loss = compute_loss(logits, labels, unk_mask, loss_type, poly_eps)

        if train:
            # The loss is a SUM over unknown labels, so accumulating micro-batches
            # reproduces the larger-batch sum exactly (no division here).
            loss.backward()
            if (batch_idx + 1) % grad_ac_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        loss_total += loss.item()
        n_seen += labels.size(0)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targs.append(labels.detach().cpu().numpy())

    return (np.concatenate(all_preds), np.concatenate(all_targs),
            loss_total / max(n_seen, 1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/storage2/cousin/datasets/MURED')
    ap.add_argument('--results_dir', default='results/mured')
    ap.add_argument('--name', default='ctran_densenet161')
    ap.add_argument('--backbone', default='densenet161',
                    choices=['densenet161', 'resnet101'])
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--grad_ac_steps', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--loss_type', default='poly', choices=['poly', 'bce'])
    ap.add_argument('--poly_eps', type=float, default=1.0)
    ap.add_argument('--use_lmt', action='store_true', default=True)
    ap.add_argument('--no_lmt', dest='use_lmt', action='store_false')
    ap.add_argument('--lp_ros', type=float, default=10.0,
                    help='LP-ROS oversampling percent (0 disables)')
    ap.add_argument('--scheduler', default='plateau', choices=['plateau', 'none'],
                    help="'none' = constant LR, which is what the paper describes "
                         "(it specifies only Adam @ 1e-5, no schedule)")
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max_train', type=int, default=-1, help='debug: cap train size')
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda'
    os.makedirs(args.results_dir, exist_ok=True)
    out_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(args.data_root, 'images', 'images')
    df_train = pd.read_csv(os.path.join(args.data_root, 'train_labels_stratified.csv'))
    df_val = pd.read_csv(os.path.join(args.data_root, 'val_labels_stratified.csv'))
    df_test = pd.read_csv(os.path.join(args.data_root, 'test_labels_stratified.csv'))
    label_cols = list(df_train.columns[1:])
    num_labels = len(label_cols)

    if args.max_train > 0:
        df_train = df_train.iloc[:args.max_train].reset_index(drop=True)
    n_before = len(df_train)
    if args.lp_ros > 0:
        df_train = lp_ros(df_train, percentage=args.lp_ros, seed=args.seed)
    print(f"Train: {n_before} -> {len(df_train)} (LP-ROS {args.lp_ros}%)   "
          f"Val: {len(df_val)}   Test: {len(df_test)}   Labels: {num_labels}")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_known = 100 if args.use_lmt else 0
    train_ds = MuredDataset(df_train, img_dir, train_tf, num_labels,
                            known_labels=train_known, testing=False)
    val_ds = MuredDataset(df_val, img_dir, eval_tf, num_labels,
                          known_labels=0, testing=True)
    test_ds = MuredDataset(df_test, img_dir, eval_tf, num_labels,
                           known_labels=0, testing=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = CTranModel(num_labels=num_labels, use_lmt=args.use_lmt, pos_emb=False,
                       layers=args.layers, heads=args.heads, dropout=args.dropout,
                       no_x_features=False, backbone=args.backbone).to(device)
    print(f"Model: C-Tran / {args.backbone}  hidden={model.backbone.out_channels}  "
          f"layers={args.layers} heads={args.heads}  params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)

    best = {'Model_Score': -1, 'epoch': 0}
    log_path = os.path.join(out_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_Model_Score,val_ML_Score,val_Bin_AUC,'
                'test_Model_Score,test_ML_Score,test_Bin_AUC,lr\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        _, _, train_loss = run_epoch(model, train_loader, optimizer, device,
                                     args.loss_type, args.poly_eps, train=True,
                                     grad_ac_steps=args.grad_ac_steps)
        val_p, val_t, val_loss = run_epoch(model, val_loader, None, device,
                                           args.loss_type, args.poly_eps, train=False)
        test_p, test_t, _ = run_epoch(model, test_loader, None, device,
                                      args.loss_type, args.poly_eps, train=False)

        vm = compute_mured_metrics(val_t, val_p, label_cols)
        tm = compute_mured_metrics(test_t, test_p, label_cols)
        lr_now = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] {dt:5.1f}s  lr={lr_now:.1e}  "
              f"train_loss={train_loss:8.2f}  val_loss={val_loss:8.2f}")
        print("  VAL ", format_metrics(vm))
        print("  TEST", format_metrics(tm))

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{vm['Model_Score']:.4f},"
                    f"{vm['ML_Score']:.4f},{vm['Bin_AUC']:.4f},{tm['Model_Score']:.4f},"
                    f"{tm['ML_Score']:.4f},{tm['Bin_AUC']:.4f},{lr_now:.2e}\n")

        if vm['Model_Score'] > best['Model_Score']:
            best = {'Model_Score': vm['Model_Score'], 'epoch': epoch,
                    'val': vm, 'test': tm}
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_metrics': vm, 'test_metrics': tm, 'args': vars(args)},
                       os.path.join(out_dir, 'best_model.pt'))
            np.savez(os.path.join(out_dir, 'best_preds.npz'),
                     val_pred=val_p, val_true=val_t, test_pred=test_p, test_true=test_t,
                     label_cols=np.array(label_cols))
            print(f"  * new best (val Model_Score={vm['Model_Score']:.4f})")

    print("\n" + "=" * 78)
    print(f"BEST @ epoch {best['epoch']}  (selected on VAL Model_Score)")
    print("-" * 78)
    print("VALIDATION:")
    print(format_metrics(best['val']))
    print("TEST:")
    print(format_metrics(best['test']))
    print("\nPaper's proposed model (Table XI): ML_F1 0.573  ML_mAP 0.685  "
          "ML_AUC 0.962  Bin_AUC 0.976  Bin_F1 0.824  Model_Score 0.900")
    print("\nPer-class (TEST) — compare to paper Table XII:")
    print(format_per_class(best['test']))
    print("=" * 78)


if __name__ == '__main__':
    main()
