#!/usr/bin/env bash
# Train + eval C-Tran/RETFound on one or more fundus datasets.
#   GPU=0 bash run_retfound.sh mured
#   GPU=0 bash run_retfound.sh mured rfmid prism
# Uses the Fiber venv (has timm + albumentations + cv2).
set -u
cd "$(dirname "$0")"
PY=/home/cousin/research/Fiber_dino/venv_fiber/bin/python
GPU="${GPU:-0}"
NAME="${NAME:-ctran_retfound_poly_bs32}"
DATASETS="${*:-mured}"

for ds in $DATASETS; do
  echo "$(date '+%F %T')  ===== TRAIN C-Tran/RETFound on $ds ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY train_retfound.py --dataset "$ds" --name "$NAME" \
    > "logs_${ds}_${NAME}.log" 2>&1
  trc=$?
  if [ $trc -ne 0 ] || [ ! -f "results/$ds/$NAME/best_preds.npz" ]; then
    echo "$(date '+%F %T')  !! $ds training failed (rc=$trc) — see logs_${ds}_${NAME}.log; skipping to next"
    continue
  fi
  # ---- evaluate + report THIS dataset immediately (before the next training) ----
  echo "$(date '+%F %T')  ===== EVAL + REPORT $ds ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_retfound.py "results/$ds/$NAME" || \
    echo "$(date '+%F %T')  !! $ds eval failed (training is still saved)"
  # echo the finished report straight into the log so it's visible right away
  rep="performance/${ds}_ctran_retfound.md"
  if [ -f "$rep" ]; then
    echo "----- $rep -----"; cat "$rep"; echo; echo "----- end $rep -----"
  fi
  echo "$(date '+%F %T')  $ds COMPLETE — report at $rep"
done
echo "$(date '+%F %T')  ALL DONE"
