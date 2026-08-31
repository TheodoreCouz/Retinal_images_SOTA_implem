#!/bin/bash
# File-based job queue for the multi-seed CI runs.
# Each worker (pinned to one GPU) atomically claims the next pending job,
# runs it to completion, then claims another. Add workers for more GPUs by
# calling `worker <gpu> &` — no restart needed.
cd /home/cousin/research/C-Tran
VENV=/home/cousin/research/Fiber/venv_fiber/bin/python
Q=results/ci_queue

claim() {
  for f in "$Q"/pending/*.job; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if mv "$f" "$Q/running/$b" 2>/dev/null; then echo "$Q/running/$b"; return 0; fi
  done
  return 1
}

worker() {
  local gpu=$1
  local job ds seed name
  while job=$(claim); do
    read ds seed < "$job"
    name=ctran_poly_constlr_bs32_seed${seed}
    echo "$(date +%H:%M:%S) GPU$gpu START $ds seed$seed" >> "$Q/progress.log"
    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $VENV -u train_fundus.py --dataset "$ds" --seed "$seed" --name "$name" \
      --workers 8 > "results/ci_${ds}_seed${seed}.out" 2>&1
    echo "$(date +%H:%M:%S) GPU$gpu DONE  $ds seed$seed (exit $?)" >> "$Q/progress.log"
    mv "$job" "$Q/done/" 2>/dev/null
  done
  echo "$(date +%H:%M:%S) GPU$gpu queue empty, worker exiting" >> "$Q/progress.log"
}

# Launch one worker per free GPU (0 and 1). Extra GPUs can be added later.
worker 0 &
worker 1 &
wait
touch "$Q/.all_done"
echo "$(date +%H:%M:%S) ALL CI RUNS COMPLETE" >> "$Q/progress.log"
