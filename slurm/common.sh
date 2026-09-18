# Shared settings for the MuReD / RFMiD SLURM runs. Sourced, not executed.
#
# Everything heavy (venvs, pretrained weights, checkpoints, logs) lives on
# /globalsc: the /home quota is ~40 GB and already nearly full.

REPO=/home/users/c/o/cousint/research/Retinal_images_SOTA_implem
STORE=/globalsc/ucl/ingi/cousint/SOTA_runs
SRC=/globalsc/ucl/ingi/cousint/ISBI_datasets

VENV=$STORE/venv
DATA=$STORE/data                 # RetExpert-layout symlink tree
OUT=$STORE/runs
PREWEIGHTS=$STORE/preweights

export MURED_DIR=$SRC/MURED      # read by C-Tran/dataloaders/fundus_prep.py
export RFMID_DIR=$SRC/RFMiD
export TORCH_HOME=$STORE/torch_home
export HF_HOME=$STORE/hf_home
export PYTHONUNBUFFERED=1
# Compute nodes have 32 cores; keep dataloader threads from oversubscribing.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

RETFOUND_CFP=/home/users/c/o/cousint/research/Fiber/RETFound/CFP/RETFound_mae_natureCFP.pth

SEEDS=(42 43 44 45 46)

mkdir -p "$OUT"

# Albumentations otherwise phones home for a version check on every import.
export NO_ALBUMENTATIONS_UPDATE=1
