#!/bin/bash
# Sequentially trains RetExpert on MuReD for 400 epochs with seeds 43-46,
# on GPU 1. Launched automatically by auto_decide_and_launch.sh only if the
# seed-42 400-epoch probe confirmed that longer training helps.
set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1

SEEDS=(43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Starting MuReD 400-epoch training with seed ${SEED}"
    echo "=================================================="
    .venv/bin/python3 train.py \
        --dataset MuReD \
        --data_path ./data/MuReD \
        --nb_classes 20 \
        --model vit_large_patch16 \
        --finetune /home/cousin/research/Fiber/RETFound/CFP/RETFound_mae_natureCFP.pth \
        --input_size 224 \
        --batch_size 16 \
        --epochs 400 \
        --lr 1e-3 \
        --weight_decay 0.05 \
        --criterion RAL \
        --adapter_mode AKU \
        --tuning_mode adapter \
        --beta_UAML 0.1 \
        --gamma_FDCM 0.1 \
        --alpha_SOA 0.1 \
        --clip_grad 1.0 \
        --num_workers 8 \
        --output_dir "./output_dir/mured_retexpert_seed${SEED}_400ep" \
        --performance_dir ./performance \
        --seed "${SEED}" \
        > "./output_dir/mured_train_seed${SEED}_400ep.log" 2>&1
    echo "Finished seed ${SEED} (400 epochs)"
done

echo "All 4 remaining MuReD 400-epoch training runs complete."
