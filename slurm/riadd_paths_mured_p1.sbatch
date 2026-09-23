#!/usr/bin/env bash
#SBATCH --job-name=riadd_p1_mured
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_p1_mured_%A_%a.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/riadd_p1_mured_%A_%a.out
#
# Path 1 (KAMATALAB's multi-stage sub-system), extended to MuReD.
#
# KAMATALAB never entered MuReD -- path 1's original sub-tasks (dr/g9/g8/g11)
# are hardcoded to RFMiD's own 29-column layout and do not transfer. This is
# OUR extension of their method, not a recovered/agreed-upon configuration:
# MuReD's 19 non-NORMAL classes are ranked by training-set frequency (the same
# rule KAMATALAB used for RFMiD) and cut into 3 tiers sized 6/5/8, apportioned
# from RFMiD's 9:8:11 ratio via largest-remainder rounding. NORMAL (MuReD's
# analogue of Disease_Risk) gets its own head, matching RFMiD's 'dr' head's
# role. See train_mured_kfold_ddp.py's MURED_SUBTASKS block for the exact
# column groups and the reasoning.
#
# One array task = one (subtask, seed) 5-fold ensemble. Paths 2+3 already
# exist for MuReD (riadd_paths_mured/p2_*, p3_*) -- this job produces only the
# missing path 1.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/Pytorch-RIADD"
mkdir -p "$OUT/logs"

PY=$VENV/riadd/bin/python
export MURED_TRAIN_CSV=$SRC/MURED/train_labels_stratified.csv
IMGDIR=$SRC/MURED/images/images
CKPT=$PREWEIGHTS/tf_efficientnet_b5_ns-6f26d0cf.pth

# "<subtask>:<seed>" x 4 subtasks x 5 seeds, ITEMS[i] is task i.
ITEMS=()
for st in normal g6 g5 g8; do
    for s in 42 43 44 45 46; do ITEMS+=("$st:$s"); done
done

IFS=: read -r subtask seed <<< "${ITEMS[$SLURM_ARRAY_TASK_ID]}"
odir="$OUT/riadd_paths_mured/p1_${subtask}_seed${seed}"

ensemble_done() {
    for f in 0 1 2 3 4; do
        compgen -G "${1}fold_${f}/train/*/model_best.pth.tar" >/dev/null || return 1
    done
    return 0
}
if ensemble_done "$odir"; then
    echo "SKIP p1/$subtask seed=$seed (all 5 folds present)"; exit 0
fi

echo "$(date '+%F %T')  ===== p1(mured) tf_efficientnet_b5_ns @768 subtask=$subtask seed=$seed ====="
$PY train_mured_kfold_ddp.py \
    --data "$IMGDIR" \
    --model tf_efficientnet_b5_ns \
    --subtask "$subtask" \
    --img-size 768 \
    -b 4 \
    -j "${SLURM_CPUS_PER_TASK:-8}" \
    --seed "$seed" \
    --initial-checkpoint "$CKPT" \
    --output "$odir" \
    > "$OUT/logs/riadd_p1_mured_${subtask}_seed${seed}.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && ensemble_done "$odir"; then
    echo "$(date '+%F %T')  DONE p1/$subtask seed=$seed"
else
    echo "$(date '+%F %T')  FAIL p1/$subtask seed=$seed (rc=$rc)"; exit 1
fi
