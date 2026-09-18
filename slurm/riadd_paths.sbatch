#!/usr/bin/env bash
#SBATCH --job-name=riadd_paths
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_paths_%A_%a.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_paths_%A_%a.out
#
# KAMATALAB's full off-site system on RFMiD (paper Appendix A.1 + Table A.11):
#
#   path one   EfficientNet-B5 @ 768, MULTI-STAGE -- sub-challenge 2 split into
#              three sub-tasks by training-set frequency (g9 / g8 / g11), plus a
#              Disease_Risk head for sub-challenge 1.
#   path two   EfficientNet-B5 @ 960, direct training (29 classes in one head)
#   path three EfficientNet-B6 @ 768, direct training
#
# The reported challenge score is the average blend of all three paths, so a
# single path cannot reach it. Our previous single B6@960 run was none of the
# three. One array task = one (path, subtask, seed) 5-fold ensemble.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/Pytorch-RIADD"
mkdir -p "$OUT/logs"

PY=$VENV/riadd/bin/python
export RIADD_TRAIN_CSV=$SRC/RFMiD/train_labels.csv
IMGDIR=$SRC/RFMiD/Training

# "<path>:<model>:<imgsize>:<subtask>" x 5 seeds, laid out so ITEMS[i] is task i.
ITEMS=()
for st in dr g9 g8 g11; do
    for s in 42 43 44 45 46; do ITEMS+=("p1:tf_efficientnet_b5_ns:768:$st:$s"); done
done
for s in 42 43 44 45 46; do ITEMS+=("p2:tf_efficientnet_b5_ns:960:full:$s"); done
for s in 42 43 44 45 46; do ITEMS+=("p3:tf_efficientnet_b6_ns:768:full:$s"); done

IFS=: read -r path model img subtask seed <<< "${ITEMS[$SLURM_ARRAY_TASK_ID]}"
case "$model" in
    *b5_ns) ckpt=$PREWEIGHTS/tf_efficientnet_b5_ns-6f26d0cf.pth ;;
    *)      ckpt=$PREWEIGHTS/tf_efficientnet_b6_ns-51548356.pth ;;
esac
odir="$OUT/riadd_paths/${path}_${subtask}_seed${seed}"

ensemble_done() {
    for f in 0 1 2 3 4; do
        compgen -G "${1}fold_${f}/train/*/model_best.pth.tar" >/dev/null || return 1
    done
    return 0
}
if ensemble_done "$odir"; then
    echo "SKIP $path/$subtask seed=$seed (all 5 folds present)"; exit 0
fi

echo "$(date '+%F %T')  ===== $path $model @${img} subtask=$subtask seed=$seed ====="
$PY train_riadd_kfold_ddp.py \
    --data "$IMGDIR" \
    --model "$model" \
    --subtask "$subtask" \
    --img-size "$img" \
    -b 4 \
    -j "${SLURM_CPUS_PER_TASK:-8}" \
    --seed "$seed" \
    --initial-checkpoint "$ckpt" \
    --output "$odir" \
    > "$OUT/logs/riadd_${path}_${subtask}_seed${seed}.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && ensemble_done "$odir"; then
    echo "$(date '+%F %T')  DONE $path/$subtask seed=$seed"
else
    echo "$(date '+%F %T')  FAIL $path/$subtask seed=$seed (rc=$rc)"; exit 1
fi
