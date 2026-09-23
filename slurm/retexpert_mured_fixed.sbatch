#!/usr/bin/env bash
#SBATCH --job-name=retexpert_mured_fixed
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/retexpert_mured_fixed_%j.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/retexpert_mured_fixed_%j.out
#
# Retrain RetExpert on MuReD (5 seeds, 200 epochs), after fixing two bugs
# found by comparing the implementation against the paper's own equations:
#   - losses/retexpert_loss.py::fdcm_loss -- Eq. 5 uses |(p_i-p_j)^2-(1-m_ij)^2|
#     summed over i<=j pairs / N; the code compared (p_i-p_j)^2 against the
#     UNSQUARED (1-m_ij) via MSE over the full C x C matrix.
#   - utils/engine.py's SOA random-block draw -- Eq. 2-3 restrict I to
#     {1,...,K-1}, excluding the final block; randint(0, len(outputlist)-1)
#     could draw the final block against itself.
# Writes to mured_seed<N>_fixedloss (not mured_seed<N>) so the original,
# pre-fix checkpoints and dumped probs are untouched for comparison.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/RetExpert"
mkdir -p "$OUT/logs" "$OUT/retexpert/performance"

PY=$VENV/retexpert/bin/python
EPOCHS=200

unset SLURM_PROCID
rc_all=0

run_one() {  # run_one <seed>
    local seed=$1
    local tag="mured_seed${seed}_fixedloss"
    local odir="$OUT/retexpert/$tag"
    if [ -f "$odir/.done" ]; then
        echo "$(date '+%F %T')  SKIP $tag (already complete)"; return 0
    fi
    mkdir -p "$odir"
    echo "$(date '+%F %T')  ===== RetExpert MuReD seed=$seed (fixed loss, ${EPOCHS} ep) ====="
    $PY train.py \
        --dataset MuReD \
        --data_path "$DATA/MuReD" \
        --nb_classes 20 \
        --model vit_large_patch16 \
        --finetune "$RETFOUND_CFP" \
        --input_size 224 \
        --batch_size 16 \
        --epochs $EPOCHS \
        --lr 1e-3 \
        --weight_decay 0.05 \
        --criterion RAL \
        --adapter_mode AKU \
        --tuning_mode adapter \
        --beta_UAML 0.1 \
        --gamma_FDCM 0.1 \
        --alpha_SOA 0.1 \
        --clip_grad 1.0 \
        --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
        --seed "$seed" \
        --output_dir "$odir" \
        --performance_dir "$OUT/retexpert/performance" \
        > "$OUT/logs/retexpert_${tag}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        touch "$odir/.done"
        echo "$(date '+%F %T')  DONE $tag"
    else
        rc_all=1
        echo "$(date '+%F %T')  FAIL $tag (rc=$rc) -- see $OUT/logs/retexpert_${tag}.log"
    fi
}

for s in "${SEEDS[@]}"; do run_one "$s"; done

echo "$(date '+%F %T')  RetExpert MuReD fixed-loss job finished (rc_all=$rc_all)"
exit $rc_all
