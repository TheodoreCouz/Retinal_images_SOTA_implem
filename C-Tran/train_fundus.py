"""
Train C-Tran (DenseNet161) on a fundus multi-label dataset (mured/rfmid/prism)
using the `ctran_poly_constlr_bs32` configuration:
  DenseNet161, Adam lr 1e-5 (constant), PolyLoss, LP-ROS 10%, LMT, 384x384,
  effective batch 32 (16 x 2 grad-accum), 80 epochs.

Checkpoint selection: best epoch by VALIDATION F1 (per-sample, 0.5 threshold =
the RetExpert protocol in metrics.txt), so the reported numbers are directly
comparable to the RetExpert per-dataset results.
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import CTranModel
from dataloaders.mured_dataset import MuredDataset
from dataloaders.resampling import lp_ros
from dataloaders.fundus_prep import PREP
from train_mured import build_transforms, run_epoch, set_seed
from utils.retexpert_metrics import compute_metrics, recall_at_k


def evaluate(probs, targs):
    m = compute_metrics(probs, targs)
    m['recall@5'] = recall_at_k(probs, targs, k=5)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['mured', 'rfmid', 'prism'])
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--name', default='ctran_poly_constlr_bs32')
    ap.add_argument('--backbone', default='densenet161')
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--grad_ac_steps', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--loss_type', default='poly', choices=['poly', 'bce'])
    ap.add_argument('--poly_eps', type=float, default=1.0)
    ap.add_argument('--use_lmt', action='store_true', default=True)
    ap.add_argument('--lp_ros', type=float, default=10.0)
    ap.add_argument('--scheduler', default='none', choices=['plateau', 'none'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max_train', type=int, default=-1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda'
    out_dir = os.path.join(args.results_dir, args.dataset, args.name)
    os.makedirs(out_dir, exist_ok=True)

    train_df, val_df, test_df, label_cols = PREP[args.dataset]()
    num_labels = len(label_cols)
    if args.max_train > 0:
        train_df = train_df.iloc[:args.max_train].reset_index(drop=True)
    n_before = len(train_df)
    if args.lp_ros > 0:
        train_df = lp_ros(train_df, percentage=args.lp_ros, seed=args.seed)
    print(f"[{args.dataset}] train {n_before}->{len(train_df)} (LP-ROS {args.lp_ros}%)  "
          f"val {len(val_df)}  test {len(test_df)}  labels {num_labels}")
    # Save the exact test set so eval can rebuild it deterministically.
    test_df.to_csv(os.path.join(out_dir, 'test_df.csv'), index=False)

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = MuredDataset(train_df, '', train_tf, num_labels,
                            known_labels=100 if args.use_lmt else 0, testing=False)
    val_ds = MuredDataset(val_df, '', eval_tf, num_labels, known_labels=0, testing=True)
    test_ds = MuredDataset(test_df, '', eval_tf, num_labels, known_labels=0, testing=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = CTranModel(num_labels=num_labels, use_lmt=args.use_lmt, pos_emb=False,
                       layers=args.layers, heads=args.heads, dropout=args.dropout,
                       no_x_features=False, backbone=args.backbone).to(device)
    print(f"C-Tran/{args.backbone} hidden={model.backbone.out_channels} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best = {'f1': -1, 'epoch': 0}
    log_path = os.path.join(out_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_f1,val_auc,val_map,test_f1,test_auc,test_map\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        _, _, train_loss = run_epoch(model, train_loader, optimizer, device,
                                     args.loss_type, args.poly_eps, train=True,
                                     grad_ac_steps=args.grad_ac_steps)
        val_p, val_t, _ = run_epoch(model, val_loader, None, device,
                                    args.loss_type, args.poly_eps, train=False)
        test_p, test_t, _ = run_epoch(model, test_loader, None, device,
                                      args.loss_type, args.poly_eps, train=False)
        vm, tm = evaluate(val_p, val_t), evaluate(test_p, test_t)
        print(f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s "
              f"train_loss={train_loss:8.2f} | val F1={vm['f1']:.4f} AUC={vm['auc']:.4f} "
              f"mAP={vm['aupr']:.4f} | test F1={tm['f1']:.4f} AUC={tm['auc']:.4f}")
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{vm['f1']:.4f},{vm['auc']:.4f},"
                    f"{vm['aupr']:.4f},{tm['f1']:.4f},{tm['auc']:.4f},{tm['aupr']:.4f}\n")

        if vm['f1'] > best['f1']:
            best = {'f1': vm['f1'], 'epoch': epoch, 'val': vm, 'test': tm}
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_metrics': vm, 'test_metrics': tm,
                        'label_cols': label_cols, 'args': vars(args)},
                       os.path.join(out_dir, 'best_model.pt'))
            np.savez(os.path.join(out_dir, 'best_preds.npz'),
                     test_pred=test_p, test_true=test_t,
                     val_pred=val_p, val_true=val_t,
                     label_cols=np.array(label_cols))
            print(f"  * new best (val F1={vm['f1']:.4f})")

    print(f"\nBEST @ epoch {best['epoch']} (val F1={best['f1']:.4f})")
    print("TEST:", {k: round(best['test'][k], 4) for k in
                    ['f1', 'aupr', 'auc', 'precision', 'recall', 'kappa', 'recall@5']})


if __name__ == '__main__':
    main()
