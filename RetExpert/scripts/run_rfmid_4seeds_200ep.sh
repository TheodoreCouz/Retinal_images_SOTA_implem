#!/bin/bash
# Sequentially trains RetExpert on RFMiD for 200 epochs with seeds 43-46,
# on GPU 3. Completes the 5-seed picture at 200 epochs to match MuReD
# (RFMiD previously only had seed 42 at 200 epochs).
set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3

SEEDS=(43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Starting RFMiD 200-epoch training with seed ${SEED}"
    echo "=================================================="
    .venv/bin/python3 train.py \
        --dataset RFMiD \
        --data_path ./data/RFMiD \
        --nb_classes 29 \
        --model vit_large_patch16 \
        --finetune /home/cousin/research/Fiber/RETFound/CFP/RETFound_mae_natureCFP.pth \
        --input_size 224 \
        --batch_size 16 \
        --epochs 200 \
        --lr 1e-3 \
        --weight_decay 0.05 \
        --criterion RAL \
        --adapter_mode AKU \
        --tuning_mode adapter \
        --beta_UAML 0.1 \
        --gamma_FDCM 0 \
        --alpha_SOA 0.1 \
        --clip_grad 1.0 \
        --num_workers 8 \
        --output_dir "./output_dir/rfmid_retexpert_seed${SEED}_200ep" \
        --performance_dir ./performance \
        --seed "${SEED}" \
        > "./output_dir/rfmid_train_seed${SEED}_200ep.log" 2>&1
    echo "Finished seed ${SEED} (200 epochs)"
done

echo "All 4 remaining RFMiD 200-epoch training runs complete."
