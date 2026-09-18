#!/usr/bin/env bash
#SBATCH --job-name=eval_table
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/eval_table_%j.out
#SBATCH --error=/globalsc/ucl/ingi/cousint/SOTA_runs/runs/logs/eval_table_%j.out
#
# Score all 30 runs under one protocol and print the comparison table.
# C-Tran already saved its test probabilities at training time; the other two
# need one inference pass each, in their own venv.
set -eu
source /home/users/c/o/cousint/research/Retinal_images_SOTA_implem/slurm/common.sh
export RUNS_DIR=$OUT SRC_DIR=$SRC DATA_DIR=$DATA
export PROBS_DIR=$STORE/probs
mkdir -p "$PROBS_DIR"

# C-Tran: reuse the npz train_fundus.py already wrote (test_pred / test_true).
for d in mured rfmid; do for s in 42 43 44 45 46; do
    ln -sfn "$OUT/ctran/$d/ctran_poly_constlr_bs32_seed$s/best_preds.npz" \
            "$PROBS_DIR/ctran_${d}_seed${s}.npz"
done; done

unset SLURM_PROCID   # RetExpert: single-process, as in the training jobs
( cd "$REPO/RetExpert"    && $VENV/retexpert/bin/python dump_test_probs.py )
( cd "$REPO/Pytorch-RIADD" && $VENV/riadd/bin/python dump_test_probs.py )

$VENV/ctran/bin/python "$REPO/make_table.py" | tee "$STORE/runs/comparison_table.md"
