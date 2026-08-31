# --------------------------------------------------------------------------
# RetExpert: A Test-time Clinically Adaptive Framework for Retinal Disease Detection
#
# Official Implementation of the Paper:
# "RetExpert: A test-time clinically adaptive framework for detecting multiple 
#  fundus diseases by harnessing ophthalmic foundation models"
#
# Authors: Hongyang Jiang, Zirong Liu, et al.
# Copyright (c) 2025 The Chinese University of Hong Kong & Wenzhou Medical University.
#
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# train.py
import argparse
import os
import time
import datetime
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path


from models.vit_adapter import build_retexpert_vit_large
from losses.retexpert_loss import RetExpertLoss
from config import get_dataset_config, set_trainable_parameters
from utils.engine import train_one_epoch, evaluate
from utils.constants import MURED_FDCM, ODIR_FDCM, MURED_FDCM, ODIR_FDCM
import utils.misc as misc
from utils.datasets import build_dataset, build_ODIR_dataset 
from losses import Ralloss

def get_args_parser():
    parser = argparse.ArgumentParser('RetExpert Training', add_help=False)
    
    # basic parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Gradient accumulation step')
    
    # model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.2)

    # augmentation parameters (used by utils.datasets.build_transform)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='AutoAugment policy')
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase probability')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--finetune', default='', help='Path to pretrained checkpoint (e.g. RETFound)')
    parser.add_argument('--tuning_mode', default='adapter', type=str,
                        help='Fine-tuning strategy: adapter, full, head, or custom')

    # RetExpert specific parameters
    parser.add_argument('--adapter_dim', default=64, type=int, help='Dimension of AKU Adapter')
    parser.add_argument('--adapter_mode', default='AKU', type=str, choices=['AKU', 'AKU_att'],
                        help='AKU=Adapter in Attn&MLP; AKU_att=Adapter in Attn only')
    
    # optimizer and lr schedule
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--layer_decay', type=float, default=0.65, help='Layer-wise lr decay')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping max-norm (set <=0 to disable clipping)')
    
    # loss function parameters
    parser.add_argument('--criterion', default='RAL', type=str, choices=['RAL', 'BCE'])
    parser.add_argument('--beta_UAML', type=float, default=0.1, help='Weight for UAML loss')
    parser.add_argument('--gamma_FDCM', type=float, default=0.1, help='Weight for FDCM loss')
    parser.add_argument('--alpha_SOA', type=float, default=0.1, help='Weight for SOA (KLD) loss')
    
    # data and environment
    parser.add_argument('--data_path', default='./data/MuReD', type=str)
    parser.add_argument('--dataset', default='MuReD', type=str, choices=['MuReD', 'ODIR', 'RFMiD', 'PRISM'])
    parser.add_argument('--nb_classes', default=20, type=int)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true', help='Use ITP-style distributed env vars')
    parser.add_argument('--performance_dir', default='./performance', type=str,
                        help='Directory to save the final test-set performance report')

    return parser

def main(args):
    # 1. initialize distributed training
    ds_config = get_dataset_config(args.dataset)
    args.nb_classes = ds_config['num_classes']
    print(f"Loaded config for {args.dataset}: {args.nb_classes} classes.")
    
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # 2. set random seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 3. build datasets
    if args.dataset == 'MuReD':
        dataset_train = build_dataset(is_train='train', args=args)
        dataset_val = build_dataset(is_train='val', args=args)
        dataset_test = build_dataset(is_train='test', args=args)
        fdcm_matrix = MURED_FDCM 
    elif args.dataset == 'ODIR':
        dataset_train = build_ODIR_dataset(is_train='train', args=args)
        dataset_val = build_ODIR_dataset(is_train='val', args=args)
        dataset_test = build_ODIR_dataset(is_train='test', args=args)
        fdcm_matrix = ODIR_FDCM
    elif args.dataset == 'RFMiD':
        dataset_train = build_dataset(is_train='train', args=args)
        dataset_val = build_dataset(is_train='val', args=args)
        dataset_test = build_dataset(is_train='test', args=args)
        fdcm_matrix = None  # No curated FDCM prior available for RFMiD; gamma_FDCM should be 0
    elif args.dataset == 'PRISM':
        dataset_train = build_dataset(is_train='train', args=args)
        dataset_val = build_dataset(is_train='val', args=args)
        dataset_test = build_dataset(is_train='test', args=args)
        fdcm_matrix = None  # No curated FDCM prior available for PRISM; gamma_FDCM should be 0
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    # distributed samplers
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    # 4. build model (RetExpert ViT Large)
    print(f"Creating model: {args.model}")
    model = build_retexpert_vit_large(
        num_classes=args.nb_classes,
        adapter_mode=args.adapter_mode,
        adapter_dim=args.adapter_dim
    )

    # load pretrained weights (e.g. RETFound)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        
        # process state_dict to remove mismatched keys
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
                del checkpoint_model[k]
        
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 5. Freeze Parameters (Freeze Backbone, Unfreeze Adapter/Norm/Head)
    # Ref: Section 2.3 of the paper
    _ = set_trainable_parameters(model_without_ddp, args.tuning_mode)
            
    # print trainable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable params: {n_parameters / 1e6:.2f}M')

    # 6. setup loss functions
    # (A) main loss function (RAL or BCE)
    if args.criterion == 'RAL':
        criterion_cls = Ralloss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8, 
                                lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0)
    else:
        criterion_cls = torch.nn.BCEWithLogitsLoss()
    
    # (B) RetExpert auxiliary loss (UAML + FDCM + SOA)
    criterion_aux = RetExpertLoss(
        num_classes=args.nb_classes,
        device=device,
        fdcm_matrix=fdcm_matrix, # FDCM matrix
        alpha_kl=3.0 # SOA temperature
    )

    # 7. optimizer
    # optimize only the trainable parameters
    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    # 8. training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_f1 = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        # training
        train_stats = train_one_epoch(
            model, criterion_cls, criterion_aux,
            data_loader_train, optimizer, device, epoch,
            loss_scaler, max_norm=(args.clip_grad if args.clip_grad > 0 else None), args=args
        )
        
        # evaluation
        if (epoch % 5 == 0) or (epoch + 1 == args.epochs):
            val_stats = evaluate(data_loader_val, model, device, args=args)
            print(f"Epoch {epoch} Val: F1={val_stats['f1']:.4f} Kappa={val_stats['kappa']:.4f}")

            # save best model
            if val_stats["f1"] > max_f1:
                max_f1 = val_stats["f1"]
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, 
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                    )
        
        # logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 9. Final evaluation on held-out test set using the best (val-selected) checkpoint
    best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.exists(best_ckpt_path) and misc.is_main_process():
        print(f"Loading best checkpoint from {best_ckpt_path} for final test evaluation")
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        test_stats = evaluate(data_loader_test, model, device, args=args)
        print(f"Final Test Results: F1={test_stats['f1']:.4f} Kappa={test_stats['kappa']:.4f} "
              f"AUC={test_stats['auc']:.4f} AUPR={test_stats['aupr']:.4f}")

        Path(args.performance_dir).mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": args.dataset,
            "model": args.model,
            "adapter_mode": args.adapter_mode,
            "tuning_mode": args.tuning_mode,
            "criterion": args.criterion,
            "beta_UAML": args.beta_UAML,
            "gamma_FDCM": args.gamma_FDCM,
            "alpha_SOA": args.alpha_SOA,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "best_val_checkpoint_epoch": checkpoint.get("epoch"),
            "test_metrics": {k: float(v) for k, v in test_stats.items()},
        }
        out_path = os.path.join(args.performance_dir, f"{args.dataset}_retexpert_seed{args.seed}_ep{args.epochs}_test_performance.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved performance report to {out_path}")

        # Markdown report (human-readable companion to the JSON above)
        md_path = os.path.join(args.performance_dir, f"{args.dataset}_retexpert_seed{args.seed}_ep{args.epochs}_test_performance.md")
        m = report["test_metrics"]
        md_lines = [
            f"# RetExpert on {args.dataset} — Test Performance",
            "",
            f"- Model: `{args.model}` (adapter_mode={args.adapter_mode}, tuning_mode={args.tuning_mode})",
            f"- Criterion: {args.criterion} (beta_UAML={args.beta_UAML}, gamma_FDCM={args.gamma_FDCM}, alpha_SOA={args.alpha_SOA})",
            f"- Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, Seed: {args.seed}",
            f"- Best checkpoint selected at epoch: {report['best_val_checkpoint_epoch']} (by validation F1)",
            "",
            "## Test set metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {m['acc']:.4f} |",
            f"| Precision | {m['precision']:.4f} |",
            f"| Recall | {m['recall']:.4f} |",
            f"| F1 score | {m['f1']:.4f} |",
            f"| AUROC | {m['auc']:.4f} |",
            f"| AUPRC | {m['aupr']:.4f} |",
            f"| Kappa | {m['kappa']:.4f} |",
            f"| Hamming loss | {m['hamming']:.4f} |",
            "",
        ]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"Saved markdown performance report to {md_path}")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)