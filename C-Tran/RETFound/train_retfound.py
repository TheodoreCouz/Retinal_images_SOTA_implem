"""Train C-Tran with a RETFound (ViT-L/16) backbone on a fundus multi-label
dataset (mured / rfmid / prism).

Same C-Tran recipe as the DenseNet run (`ctran_poly_constlr_bs32`): Adam, constant
LR, PolyLoss, LP-ROS 10%, LMT, 384x384, effective batch 32, best epoch by
validation F1 — so numbers are directly comparable to the DenseNet C-Tran and
the RetExpert per-dataset results. Only the encoder changes: the backbone is
initialised from `best_retfound_lse_select[_<ds>]` (the fundus-fine-tuned
RETFound), bf16 autocast + gradient checkpointing keep ViT-L@384 in memory, and
the backbone is frozen for the first `--freeze_epochs` so the fresh C-Tran head
warms up before the encoder is unfrozen.

  cd C-Tran/RETFound
  CUDA_VISIBLE_DEVICES=0 <fiber-venv>/bin/python train_retfound.py --dataset mured
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_CTRAN_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _CTRAN_ROOT)
sys.path.insert(0, _HERE)

from dataloaders.mured_dataset import MuredDataset            # noqa: E402
from dataloaders.resampling import lp_ros                     # noqa: E402
from dataloaders.fundus_prep import PREP                      # noqa: E402
from train_mured import build_transforms, set_seed, compute_loss  # noqa: E402
from models.utils import custom_replace                       # noqa: E402
from utils.retexpert_metrics import compute_metrics, recall_at_k  # noqa: E402
from ctran_retfound import CTranModelRETFound                 # noqa: E402

# Per-dataset RETFound-LSE fine-tuned encoder (the "best_retfound_lse_select" run).
FIBER = "/home/cousin/research/Fiber_dino/models"
RETFOUND_CKPT = {
    "mured": f"{FIBER}/best_retfound_lse_select_mured/weights/sign_classifier.pt",
    "rfmid": f"{FIBER}/best_retfound_lse_select_rfmid/weights/sign_classifier.pt",
    "prism": f"{FIBER}/best_retfound_lse_select/weights/sign_classifier.pt",
}


def evaluate(probs, targs):
    m = compute_metrics(probs, targs)
    m["recall@5"] = recall_at_k(probs, targs, k=5)
    return m


def run_epoch(model, loader, optimizer, device, loss_type, poly_eps,
              train, grad_ac_steps=1, amp=True):
    """One epoch. bf16 autocast + grad-accum. Returns sigmoid preds, targets, avg loss."""
    model.train() if train else model.eval()
    if train:
        optimizer.zero_grad()
    all_preds, all_targs, loss_total, n_seen = [], [], 0.0, 0
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask = batch["mask"].float().to(device)
        unk_mask = custom_replace(mask, 1, 0, 0)               # 1 at unknown positions
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _, _ = model(images, mask.clone())
            logits = logits.view(labels.size(0), -1).float()
            loss = compute_loss(logits, labels, unk_mask, loss_type, poly_eps)
        if train:
            loss.backward()
            if (batch_idx + 1) % grad_ac_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        loss_total += loss.item()
        n_seen += labels.size(0)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targs.append(labels.detach().cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targs), loss_total / max(n_seen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mured", "rfmid", "prism"])
    ap.add_argument("--results_dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--name", default="ctran_retfound_poly_bs32")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_ac_steps", type=int, default=4)      # effective batch 32
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--drop_path_rate", type=float, default=0.2)
    ap.add_argument("--loss_type", default="poly", choices=["poly", "bce"])
    ap.add_argument("--poly_eps", type=float, default=1.0)
    ap.add_argument("--use_lmt", action="store_true", default=True)
    ap.add_argument("--lp_ros", type=float, default=10.0)
    ap.add_argument("--freeze_epochs", type=int, default=3,
                    help="freeze the ViT backbone for the first N epochs (head warm-up)")
    ap.add_argument("--retfound_weights", default=None,
                    help="override the per-dataset best_retfound_lse_select checkpoint")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=-1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    weights = args.retfound_weights or RETFOUND_CKPT[args.dataset]
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
    test_df.to_csv(os.path.join(out_dir, "test_df.csv"), index=False)

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = MuredDataset(train_df, "", train_tf, num_labels,
                            known_labels=100 if args.use_lmt else 0, testing=False)
    val_ds = MuredDataset(val_df, "", eval_tf, num_labels, known_labels=0, testing=True)
    test_ds = MuredDataset(test_df, "", eval_tf, num_labels, known_labels=0, testing=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = CTranModelRETFound(
        num_labels=num_labels, use_lmt=args.use_lmt, layers=args.layers,
        heads=args.heads, dropout=args.dropout, retfound_weights=weights,
        img_size=args.img_size, drop_path_rate=args.drop_path_rate,
        grad_checkpointing=True).to(device)
    print(f"C-Tran/RETFound hidden={model.backbone.out_channels} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"backbone={os.path.relpath(weights, FIBER)}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    amp = not args.no_amp

    best = {"f1": -1, "epoch": 0}
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_auc,val_map,test_f1,test_auc,test_map\n")

    frozen = False
    for epoch in range(1, args.epochs + 1):
        # backbone warm-up: freeze ViT for the first freeze_epochs, then unfreeze
        if epoch <= args.freeze_epochs and not frozen:
            model.backbone.freeze(); frozen = True
            print(f"  [freeze] backbone frozen (epochs 1..{args.freeze_epochs})")
        if epoch == args.freeze_epochs + 1 and frozen:
            model.backbone.unfreeze(); frozen = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            print("  [freeze] backbone unfrozen")

        t0 = time.time()
        _, _, train_loss = run_epoch(model, train_loader, optimizer, device,
                                     args.loss_type, args.poly_eps, train=True,
                                     grad_ac_steps=args.grad_ac_steps, amp=amp)
        val_p, val_t, _ = run_epoch(model, val_loader, None, device,
                                    args.loss_type, args.poly_eps, train=False, amp=amp)
        test_p, test_t, _ = run_epoch(model, test_loader, None, device,
                                      args.loss_type, args.poly_eps, train=False, amp=amp)
        vm, tm = evaluate(val_p, val_t), evaluate(test_p, test_t)
        print(f"[{epoch:3d}/{args.epochs}] {time.time()-t0:5.1f}s "
              f"train_loss={train_loss:8.2f} | val F1={vm['f1']:.4f} AUC={vm['auc']:.4f} "
              f"mAP={vm['aupr']:.4f} | test F1={tm['f1']:.4f} AUC={tm['auc']:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{vm['f1']:.4f},{vm['auc']:.4f},"
                    f"{vm['aupr']:.4f},{tm['f1']:.4f},{tm['auc']:.4f},{tm['aupr']:.4f}\n")

        if vm["f1"] > best["f1"]:
            best = {"f1": vm["f1"], "epoch": epoch, "val": vm, "test": tm}
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_metrics": vm, "test_metrics": tm,
                        "label_cols": label_cols, "args": vars(args),
                        "backbone": "retfound"},
                       os.path.join(out_dir, "best_model.pt"))
            np.savez(os.path.join(out_dir, "best_preds.npz"),
                     test_pred=test_p, test_true=test_t, val_pred=val_p, val_true=val_t,
                     label_cols=np.array(label_cols))
            print(f"  * new best (val F1={vm['f1']:.4f})")

    print(f"\nBEST @ epoch {best['epoch']} (val F1={best['f1']:.4f})")
    print("TEST:", {k: round(best["test"][k], 4) for k in
                    ["f1", "aupr", "auc", "precision", "recall", "kappa", "recall@5"]})


if __name__ == "__main__":
    main()
