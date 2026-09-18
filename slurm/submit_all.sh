#!/usr/bin/env bash
# Submit all 30 trainings: 3 methods x 2 datasets (MuReD, RFMiD) x 5 seeds.
#
#   RetExpert     1 job  x 10 runs   (sequential on one GPU)
#   C-Tran        3 jobs x 3-4 runs  (array 0-2, split 4/3/3)
#   Pytorch-RIADD 5 jobs x 2 runs    (array 0-4; one run = one 5-fold ensemble)
#
# Every job is idempotent: re-submitting skips work that already completed, so
# a job that hits its wall clock can simply be resubmitted.
set -euo pipefail
cd "$(dirname "$0")"
sbatch retexpert.sbatch
sbatch ctran.sbatch
sbatch riadd.sbatch
