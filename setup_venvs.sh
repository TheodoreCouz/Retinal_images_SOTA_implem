#!/usr/bin/env bash
# Build the three per-method virtualenvs used by the MuReD / RFMiD runs.
#
# Why three and not one: the methods have mutually incompatible pins
# (see README "Environments").
#   - Pytorch-RIADD vendors timm 0.4.3, whose riadd_augment.py needs the
#     albumentations 0.5.x API (IAAAdditiveGaussianNoise, Cutout), both removed
#     in albumentations 1.x.
#   - C-Tran's train_mured.build_transforms uses the albumentations 2.x API
#     (GaussNoise(std_range=), CoarseDropout(num_holes_range=, fill=)).
#   - RetExpert needs a modern timm (timm.models.vision_transformer.Block).
# All three share torch cu121 wheels: the cluster GPU is an RTX 6000 Ada
# (sm_89), which the repos' original torch 1.8 / 1.12+cu113 pins cannot target.
#
# Where: the /home quota (~40 GB) is nearly full, so the venvs live on /globalsc
# (mounted on the compute nodes, same filesystem as the datasets) and ./venv is
# a symlink to them.
set -euo pipefail
cd "$(dirname "$0")"

VENV_STORE=/globalsc/ucl/ingi/cousint/SOTA_runs/venv
mkdir -p "$VENV_STORE"
ln -sfn "$VENV_STORE" venv

export PIP_CACHE_DIR=/globalsc/ucl/ingi/cousint/SOTA_runs/pip-cache
PY=python3.9
TORCH_ARGS=(torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121)

mk() {  # mk <name>
  rm -rf "$VENV_STORE/$1"
  $PY -m venv "$VENV_STORE/$1"
  "$VENV_STORE/$1/bin/pip" install -q --upgrade pip setuptools wheel
}

echo "=== [1/3] retexpert ==="
mk retexpert
"$VENV_STORE"/retexpert/bin/pip install -q "${TORCH_ARGS[@]}"
"$VENV_STORE"/retexpert/bin/pip install -q \
  "timm==1.0.11" "numpy<2" pandas scikit-learn opencv-python-headless Pillow \
  pycm tensorboard openpyxl matplotlib

echo "=== [2/3] ctran ==="
mk ctran
"$VENV_STORE"/ctran/bin/pip install -q "${TORCH_ARGS[@]}"
"$VENV_STORE"/ctran/bin/pip install -q \
  "albumentations==2.0.5" "numpy<2" pandas scikit-learn scikit-image scipy \
  opencv-python-headless Pillow tqdm matplotlib "timm==1.0.11"

echo "=== [3/3] riadd ==="
mk riadd
"$VENV_STORE"/riadd/bin/pip install -q "${TORCH_ARGS[@]}"
# albumentations 0.5.2 predates modern build metadata, so install its runtime
# deps explicitly and then the package itself with --no-deps.
"$VENV_STORE"/riadd/bin/pip install -q \
  "numpy==1.26.4" "scipy<1.14" "scikit-image==0.22.0" "imgaug==0.4.0" \
  opencv-python-headless Pillow pyyaml pandas scikit-learn tqdm matplotlib \
  tensorboard ml_collections
# timm/utils/vis.py imports visdom unconditionally. visdom 0.2.4 is sdist-only
# and its setup.py needs pkg_resources, which setuptools>=81 no longer exposes
# in an isolated build env -- so build it against the venv's own setuptools.
"$VENV_STORE"/riadd/bin/pip install -q "setuptools<81"
"$VENV_STORE"/riadd/bin/pip install -q --no-build-isolation visdom
"$VENV_STORE"/riadd/bin/pip install -q --no-deps "albumentations==0.5.2"

echo "=== done ==="
