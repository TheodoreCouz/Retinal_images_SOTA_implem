#!/usr/bin/env bash
#SBATCH --job-name=riadd_p23_mured
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/logs/riadd_p23_mured_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=ly-w213
#SBATCH --array=0-9
#
# KAMATALAB paths 2 and 3 on MuReD -- the configuration the winner was RANKED
# with (challenge paper, Appendix A.1: the on-site result used paths two and
# three only). Path 1 is not run: its three class-frequency sub-tasks and its
# Disease_Risk head are hardcoded to RFMiD's label set and do not exist for
# MuReD, and we do not modify their method.
#   path two   EfficientNet-B5 @ 960, direct 20-class training, 5-fold
#   path three EfficientNet-B6 @ 768, direct 20-class training, 5-fold
# One array task = one (path, seed) 5-fold ensemble. Same script, batch size and
# ImageNet-NS checkpoints as the existing MuReD single-path run.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/Pytorch-RIADD"
mkdir -p "$OUT/logs" "$OUT/riadd_paths_mured"
PY=$VENV/riadd/bin/python
export MURED_TRAIN_CSV=$SRC/MURED/train_labels_stratified.csv
IMGDIR=$SRC/MURED/images/images

ITEMS=()
for s in 42 43 44 45 46; do ITEMS+=("p2:tf_efficientnet_b5_ns:960:$s"); done
for s in 42 43 44 45 46; do ITEMS+=("p3:tf_efficientnet_b6_ns:768:$s"); done
IFS=: read -r path model img seed <<< "${ITEMS[$SLURM_ARRAY_TASK_ID]}"
case "$model" in
  tf_efficientnet_b5_ns) ckpt=$PREWEIGHTS/tf_efficientnet_b5_ns-6f26d0cf.pth ;;
  tf_efficientnet_b6_ns) ckpt=$PREWEIGHTS/tf_efficientnet_b6_ns-51548356.pth ;;
esac
odir="$OUT/riadd_paths_mured/${path}_seed${seed}"

ensemble_done() {
    for f in 0 1 2 3 4; do
        compgen -G "${1}fold_${f}/train/*/model_best.pth.tar" >/dev/null || return 1
    done
    return 0
}
if ensemble_done "$odir"; then echo "SKIP $path seed=$seed (all 5 folds present)"; exit 0; fi

echo "$(date '+%F %T')  ===== MuReD $path $model @${img} seed=$seed (5 folds) ====="
$PY train_mured_kfold_ddp.py \
    --data "$IMGDIR" \
    --model "$model" \
    --num-classes 20 \
    --img-size "$img" \
    -b 4 \
    -j "${SLURM_CPUS_PER_TASK:-8}" \
    --seed "$seed" \
    --initial-checkpoint "$ckpt" \
    --output "$odir" \
    > "$OUT/logs/riadd_mured_${path}_seed${seed}.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && ensemble_done "$odir"; then
    echo "$(date '+%F %T')  DONE $path seed=$seed"
else
    echo "$(date '+%F %T')  FAIL $path seed=$seed (rc=$rc) -- see $OUT/logs/riadd_mured_${path}_seed${seed}.log"; exit 1
fi
