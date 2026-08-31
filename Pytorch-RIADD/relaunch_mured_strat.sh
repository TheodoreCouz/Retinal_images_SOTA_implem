#!/usr/bin/env bash
# Relaunch RIADD MURED training from scratch on the stratified split, GPUs 2,3.
# Kills any lingering run, clears checkpoints, waits for the GPUs to actually
# free (avoids startup OOM), then trains all 5 seeds sequentially with retry.
set -u
cd /home/cousin/research/Pytorch-RIADD

PY=.venv/bin/python
GPUS=2,3
PORT=29510
CSV=/storage2/cousin/datasets/MURED/train_labels_stratified.csv
PREFIX=ckpt/tf_efficientnet_b6_ns-MURED-STRAT-SGD1E-1-seed
LOGDIR=performance/ci_runs/logs

# 1. Kill any lingering MURED-STRAT training processes (launcher + DDP workers)
pkill -9 -f 'train_mured_kfold_ddp.py' || true
sleep 5

# 2. From scratch: remove all MURED-STRAT checkpoints
rm -rf ${PREFIX}*

# 3. Wait until GPUs 2 AND 3 are actually free (<1 GB used) before starting
echo "waiting for GPUs 2,3 to free..."
while :; do
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
  u3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  if [ "$u2" -lt 1000 ] && [ "$u3" -lt 1000 ]; then break; fi
  sleep 3
done
echo "GPUs free -> launching"

# 4. Train all 5 seeds sequentially on GPUs 2,3 (2 attempts each)
mkdir -p "$LOGDIR"
for S in 42 43 44 45 46; do
  for attempt in 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUS MURED_TRAIN_CSV=$CSV \
    $PY -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
      train_mured_kfold_ddp.py --img-size 960 -b 2 --seed "$S" \
      --output "${PREFIX}${S}" \
      > "$LOGDIR/mured_strat_seed${S}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then break; fi
    echo "seed $S attempt $attempt failed (rc=$rc) - retrying after 15s"
    sleep 15
  done
done
echo "all seeds done"
