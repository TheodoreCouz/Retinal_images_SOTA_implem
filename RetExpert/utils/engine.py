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

# utils/engine.py
import math
import sys
import random
import torch
import numpy as np
from typing import Iterable, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, cohen_kappa_score, average_precision_score, hamming_loss
)
import utils.misc as misc
import utils.lr_sched as lr_sched

def _safe_macro_score(score_fn, targets, probs):
    """
    Per-class macro average that skips degenerate columns (a class with only
    positives or only negatives across the whole set has an undefined AUC/AUPRC).
    This avoids NaN propagation that `average='samples'` suffers from when a
    *sample* (not class) has an all-zero row (common in datasets with many
    healthy/no-finding images, e.g. RFMiD), since sample-wise AUC additionally
    requires each row to contain both a positive and a negative label.
    """
    scores = []
    for c in range(targets.shape[1]):
        yt = targets[:, c]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(score_fn(yt, probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(probs, targets, threshold=0.5, average='samples'):
    """
    Compute comprehensive metrics for multi-label classification.
    Args:
        probs: Predicted probabilities (N, C)
        targets: Ground truth labels (N, C)
        threshold: Decision threshold
        average: Averaging strategy for precision/recall/F1 ('samples', 'macro', etc.)
                 AUROC/AUPRC always use a safe per-class macro average (see _safe_macro_score).
    """
    preds = (probs >= threshold).astype(int)
    targets = targets.astype(int)

    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)

    # AUROC/AUPRC: macro (per-class) average, robust to samples/classes with no positives
    auc = _safe_macro_score(roc_auc_score, targets, probs)
    aupr = _safe_macro_score(average_precision_score, targets, probs)

    # Kappa is typically for multi-class, for multi-label we often flatten or calculate per sample
    # Here we calculate based on flattened arrays to align with typical multi-label eval
    kappa = cohen_kappa_score(targets.flatten(), preds.flatten())
    hamming = hamming_loss(targets, preds)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "aupr": aupr,
        "kappa": kappa,
        "hamming": hamming
    }

def train_one_epoch(model: torch.nn.Module, 
                    criterion_cls: torch.nn.Module, # Main classification loss (e.g., RAL)
                    criterion_aux: torch.nn.Module, # RetExpert auxiliary loss
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler, 
                    max_norm: float = None,
                    mixup_fn: Optional[object] = None, 
                    log_writer=None,
                    args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per-iteration lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # Forward pass (fp16 here is safe/fast; only the matmul-heavy backbone runs in this block)
            # outputlist contains intermediate features for SOA strategy
            outputs, outputlist = model(samples)

            # --- Core Strategy: SOA (Stochastic One-hot Activation) ---
            # Randomly select one intermediate block's output
            block_num = random.randint(0, len(outputlist) - 1)
            outputs_rb = outputlist[block_num]

        # The custom RAL/UAML/FDCM/SOA losses use log/digamma/lgamma/pow, which are
        # numerically fragile in fp16 (e.g. sigmoid can hard-saturate to exact 0.0/1.0,
        # making eps-clamping ineffective and log(0) = -inf, eventually cascading to NaN).
        # Compute the loss in fp32 regardless of the forward pass's precision.
        outputs = outputs.float()
        outputs_rb = outputs_rb.float()

        # 1. Main Classification Loss (e.g., RAL or BCE)
        loss_cls = criterion_cls(outputs, targets)

        # 2. Auxiliary Losses (UAML + FDCM + SOA)
        # Note: criterion_aux is our RetExpertLoss
        loss_uaml, loss_fdcm, loss_soa, _ = criterion_aux(
            outputs, targets, epoch, args.epochs, outputs_rb
        )

        # 3. Total Loss Aggregation
        # Formula: L_total = L_cls + beta * L_uaml + gamma * L_fdcm + alpha * L_soa
        loss = loss_cls + \
               args.beta_UAML * loss_uaml + \
               args.gamma_FDCM * loss_fdcm + \
               args.alpha_SOA * loss_soa

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Logging component losses for debugging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/loss_cls', loss_cls.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_uaml', loss_uaml.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_fdcm', loss_fdcm.item(), epoch_1000x)
            log_writer.add_scalar('train/loss_soa', loss_soa.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Lists to store all predictions and targets
    all_probs = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, _ = model(images)
            
        probs = torch.sigmoid(output)
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    metrics = compute_metrics(all_probs, all_targets, threshold=0.5)
    
    print(f"Evaluation Results:")
    print(f"F1-Score: {metrics['f1']:.4f} | Kappa: {metrics['kappa']:.4f} | AUC: {metrics['auc']:.4f}")
    
    return metrics