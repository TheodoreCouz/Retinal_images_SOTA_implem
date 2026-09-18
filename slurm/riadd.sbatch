#!/usr/bin/env bash
#SBATCH --job-name=riadd
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_%A_%a.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_%A_%a.out
#
# Pytorch-RIADD (EfficientNet-B6 NoisyStudent, 960px, 20 epochs) on MuReD and
# RFMiD, 5 seeds each. One "training" = one KFold invocation that trains all 5
# folds internally (the run_ci_ensembles.sh protocol), so each array task runs
# 2 seeds = 10 fold-models.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/Pytorch-RIADD"
mkdir -p "$OUT/logs"

PY=$VENV/riadd/bin/python
IMG=960
# The upstream launchers run this under `torch.distributed.launch
# --nproc_per_node=2` at -b 2, i.e. an effective batch of 4 with --sync-bn
# active. This cluster has one GPU per node, so a single process at -b 2 halved
# the BatchNorm batch to 2 and silently dropped SyncBN (it is gated on
# args.distributed) -- which cost EfficientNet-B6 7-13 points on every metric.
# -b 4 on one GPU restores the original numerics exactly: BN over 4 samples,
# 301 optimizer steps/epoch, same LR (the script never scales LR by world size).
BS=4

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ITEMS=(mured:42 mured:43) ;;
  1) ITEMS=(mured:44 mured:45) ;;
  2) ITEMS=(mured:46 rfmid:42) ;;
  3) ITEMS=(rfmid:43 rfmid:44) ;;
  4) ITEMS=(rfmid:45 rfmid:46) ;;
  *) echo "unexpected array task ${SLURM_ARRAY_TASK_ID:-?}"; exit 2 ;;
esac

ensemble_done() {  # ensemble_done <output-prefix> -> 0 if all 5 folds saved a best model
    local out=$1
    for f in 0 1 2 3 4; do
        compgen -G "${out}fold_${f}/train/*/model_best.pth.tar" >/dev/null || return 1
    done
    return 0
}

rc_all=0
for item in "${ITEMS[@]}"; do
    ds=${item%%:*}; seed=${item##*:}
    odir="$OUT/riadd/${ds}_seed${seed}"
    if ensemble_done "$odir"; then
        echo "$(date '+%F %T')  SKIP $ds seed=$seed (all 5 folds already complete)"; continue
    fi
    echo "$(date '+%F %T')  ===== Pytorch-RIADD $ds seed=$seed (5 folds) ====="
    if [ "$ds" = mured ]; then
        script=train_mured_kfold_ddp.py
        export MURED_TRAIN_CSV=$SRC/MURED/train_labels_stratified.csv
        img_dir=$SRC/MURED/images/images
        nc=20
    else
        script=train_riadd_kfold_ddp.py
        export RIADD_TRAIN_CSV=$SRC/RFMiD/train_labels.csv
        img_dir=$SRC/RFMiD/Training
        nc=29
    fi
    $PY "$script" \
        --data "$img_dir" \
        --num-classes "$nc" \
        --img-size $IMG \
        -b $BS \
        -j "${SLURM_CPUS_PER_TASK:-8}" \
        --seed "$seed" \
        --initial-checkpoint "$PREWEIGHTS/tf_efficientnet_b6_ns-51548356.pth" \
        --output "$odir" \
        > "$OUT/logs/riadd_${ds}_seed${seed}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && ensemble_done "$odir"; then
        echo "$(date '+%F %T')  DONE $ds seed=$seed"
    else
        rc_all=1
        echo "$(date '+%F %T')  FAIL $ds seed=$seed (rc=$rc) -- see $OUT/logs/riadd_${ds}_seed${seed}.log"
    fi
done

echo "$(date '+%F %T')  Pytorch-RIADD array task ${SLURM_ARRAY_TASK_ID:-0} finished (rc_all=$rc_all)"
exit $rc_all
