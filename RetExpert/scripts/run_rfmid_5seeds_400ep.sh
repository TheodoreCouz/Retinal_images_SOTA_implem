#!/bin/bash
# Sequentially trains RetExpert on RFMiD for 400 epochs with 5 different
# seeds, on GPU 3, mirroring the MuReD 400-epoch run for comparison.
set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Starting RFMiD 400-epoch training with seed ${SEED}"
    echo "=================================================="
    .venv/bin/python3 train.py \
        --dataset RFMiD \
        --data_path ./data/RFMiD \
        --nb_classes 29 \
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
        --gamma_FDCM 0 \
        --alpha_SOA 0.1 \
        --clip_grad 1.0 \
        --num_workers 8 \
        --output_dir "./output_dir/rfmid_retexpert_seed${SEED}_400ep" \
        --performance_dir ./performance \
        --seed "${SEED}" \
        > "./output_dir/rfmid_train_seed${SEED}_400ep.log" 2>&1
    echo "Finished seed ${SEED} (400 epochs)"
done

echo "All 5 RFMiD 400-epoch training runs complete."
