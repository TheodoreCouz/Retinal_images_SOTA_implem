#!/usr/bin/env bash
# Orchestrate the 5-fold-ensemble CI training job on a SINGLE 2-GPU lane.
# 3 datasets x 5 seeds = 15 ensemble jobs, run strictly sequentially (each job is
# one KFold script invocation that trains all 5 folds internally, on 2 GPUs via DDP).
# Order: shortest dataset first (mured, rfmid, prism) so each dataset's CI is
# available as early as possible. Idempotent: completed ensembles are skipped.
set -u
cd /home/cousin/research/Pytorch-RIADD

PY=.venv/bin/python
GPUS="${CI_GPUS:-0,1}"          # the 2 GPUs to use (override with CI_GPUS=2,3)
PORT="${CI_PORT:-29500}"
SEEDS=(42 43 44 45 46)
IMG=960
BS=2
RUN_DIR=performance/ci_runs
LOG_DIR=$RUN_DIR/logs
STATUS=$RUN_DIR/status.txt
mkdir -p "$LOG_DIR"

prefix() {  # dataset -> "script output-prefix"
  case "$1" in
    mured) echo "train_mured_kfold_ddp.py ckpt/tf_efficientnet_b6_ns-MURED-KFOLD-SGD1E-1-seed" ;;
    rfmid) echo "train_riadd_kfold_ddp.py ckpt/tf_efficientnet_b6_ns-RIADD-KFOLD-SGD1E-1-seed" ;;
    prism) echo "train_prism_kfold_ddp.py ckpt/tf_efficientnet_b6_ns-PRISM-KFOLD-SGD1E-1-seed" ;;
  esac
}

ensemble_done() {  # dataset seed -> 0 if all 5 folds have a model_best
  local ds=$1 s=$2; read -r _script out <<<"$(prefix "$ds")"
  for f in 0 1 2 3 4; do
    compgen -G "${out}${s}fold_${f}/train/*/model_best.pth.tar" >/dev/null || return 1
  done
  return 0
}

log_status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

seeds_for() {  # PRISM: single run (no CI). MURED/RFMiD: 5 runs for a 95% CI.
  case "$1" in prism) echo 42 ;; *) echo "${SEEDS[*]}" ;; esac
}

log_status "orchestrator START (single lane, GPUs $GPUS): mured x5, rfmid x5, prism x1"
for ds in mured rfmid prism; do
  read -r script out <<<"$(prefix "$ds")"
  for s in $(seeds_for "$ds"); do
    outdir="${out}${s}"; logf="$LOG_DIR/${ds}_seed${s}.log"
    if ensemble_done "$ds" "$s"; then
      log_status "SKIP $ds seed=$s (already complete)"; continue
    fi
    log_status "START $ds seed=$s -> $logf"
    CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.launch \
      --nproc_per_node=2 --master_port=$PORT \
      "$script" --img-size $IMG -b $BS --seed "$s" --output "$outdir" \
      > "$logf" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && ensemble_done "$ds" "$s"; then
      log_status "DONE  $ds seed=$s (rc=0)"
    else
      log_status "FAIL  $ds seed=$s (rc=$rc) -- see $logf"
    fi
  done
done
log_status "orchestrator COMPLETE"
