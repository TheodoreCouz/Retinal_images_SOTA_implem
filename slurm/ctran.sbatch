#!/usr/bin/env bash
#SBATCH --job-name=ctran
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/ctran_%A_%a.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/ctran_%A_%a.out
#
# C-Tran (DenseNet-161, 384px, PolyLoss + LP-ROS 10% + LMT, 80 epochs) on
# MuReD and RFMiD, 5 seeds each. 10 runs split 4/3/3 over a 3-task array.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/C-Tran"
mkdir -p "$OUT/logs"

PY=$VENV/ctran/bin/python

# "<dataset>:<seed>" work items, 3-4 per array task.
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ITEMS=(mured:42 mured:43 mured:44 mured:45) ;;
  1) ITEMS=(mured:46 rfmid:42 rfmid:43) ;;
  2) ITEMS=(rfmid:44 rfmid:45 rfmid:46) ;;
  *) echo "unexpected array task ${SLURM_ARRAY_TASK_ID:-?}"; exit 2 ;;
esac

rc_all=0
for item in "${ITEMS[@]}"; do
    ds=${item%%:*}; seed=${item##*:}
    name="ctran_poly_constlr_bs32_seed${seed}"
    odir="$OUT/ctran/$ds/$name"
    if [ -f "$odir/best_preds.npz" ] && [ -f "$odir/.done" ]; then
        echo "$(date '+%F %T')  SKIP $ds seed=$seed (already complete)"; continue
    fi
    echo "$(date '+%F %T')  ===== C-Tran $ds seed=$seed ====="
    $PY train_fundus.py \
        --dataset "$ds" \
        --backbone densenet161 \
        --img_size 384 \
        --batch_size 16 \
        --grad_ac_steps 2 \
        --lr 1e-5 \
        --epochs 80 \
        --loss_type poly \
        --lp_ros 10 \
        --workers "${SLURM_CPUS_PER_TASK:-8}" \
        --seed "$seed" \
        --name "$name" \
        --results_dir "$OUT/ctran" \
        > "$OUT/logs/ctran_${ds}_seed${seed}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$odir/best_preds.npz" ]; then
        touch "$odir/.done"
        echo "$(date '+%F %T')  DONE $ds seed=$seed"
    else
        rc_all=1
        echo "$(date '+%F %T')  FAIL $ds seed=$seed (rc=$rc) -- see $OUT/logs/ctran_${ds}_seed${seed}.log"
    fi
done

echo "$(date '+%F %T')  C-Tran array task ${SLURM_ARRAY_TASK_ID:-0} finished (rc_all=$rc_all)"
exit $rc_all
