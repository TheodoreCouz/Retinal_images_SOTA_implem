#!/usr/bin/env bash
#SBATCH --job-name=retexpert_all
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/retexpert_all_%j.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/retexpert_all_%j.out
#
# RetExpert: all 10 runs (MuReD x 5 seeds, then RFMiD x 5 seeds) in one job,
# sequentially on the single GPU. 200 epochs, the config from
# scripts/run_mured_5seeds.sh and scripts/run_rfmid_4seeds_200ep.sh.
set -u
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
cd "$REPO/RetExpert"
mkdir -p "$OUT/logs" "$OUT/retexpert/performance"

PY=$VENV/retexpert/bin/python
EPOCHS=200

# Run single-process, exactly like the authors' scripts/run_mured_5seeds.sh
# (plain `python train.py` on one GPU, no launcher). Two reasons this has to be
# forced rather than left to autodetect:
#   - utils/misc.init_distributed_mode has a SLURM branch that never sets
#     args.world_size, and SLURM_PROCID is present inside a job, so it would
#     crash on AttributeError;
#   - with tuning_mode=adapter the unused att_adapter parameters make plain DDP
#     abort with "Expected to have finished reduction in the prior iteration".
# Clearing SLURM_PROCID makes init_distributed_mode take its "Not using
# distributed mode" path.
unset SLURM_PROCID
rc_all=0

run_one() {  # run_one <dataset> <nb_classes> <gamma_FDCM> <seed>
    local ds=$1 nc=$2 gamma=$3 seed=$4
    local tag="${ds,,}_seed${seed}"
    local odir="$OUT/retexpert/$tag"
    if [ -f "$odir/.done" ]; then
        echo "$(date '+%F %T')  SKIP $tag (already complete)"; return 0
    fi
    mkdir -p "$odir"
    echo "$(date '+%F %T')  ===== RetExpert $ds seed=$seed (${EPOCHS} ep) ====="
    $PY train.py \
        --dataset "$ds" \
        --data_path "$DATA/$ds" \
        --nb_classes "$nc" \
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
        --gamma_FDCM "$gamma" \
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

# MuReD uses the curated co-occurrence prior (gamma_FDCM 0.1); RFMiD has none,
# so its FDCM term is switched off (config.py: fdcm_key = None).
for s in "${SEEDS[@]}"; do run_one MuReD 20 0.1 "$s"; done
for s in "${SEEDS[@]}"; do run_one RFMiD 29 0   "$s"; done

echo "$(date '+%F %T')  RetExpert job finished (rc_all=$rc_all)"
exit $rc_all
