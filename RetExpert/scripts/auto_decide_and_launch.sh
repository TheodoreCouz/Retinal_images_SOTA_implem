#!/bin/bash
# Autonomous decision pipeline for the "does longer training close the gap
# to the paper?" hypothesis.
#
# 1. Waits for the seed-42, 400-epoch probe run (PID given as $1) to finish.
# 2. Compares its F1/Kappa against the original 200-epoch seed-42 result.
# 3. CONFIRMED iff F1 improves by >= 0.01 AND Kappa improves by >= 0.01
#    (both metrics, non-trivial margin -- roughly 1.25x the run-to-run std
#    of ~0.008 we measured across the original 5 seeds, to avoid mistaking
#    noise for a real effect).
# 4. If confirmed: launches run_mured_4seeds_400ep.sh (seeds 43-46, 400
#    epochs each, sequential, GPU 1) in the background, unattended.
# 5. Writes a full decision log either way, so the outcome is auditable
#    even if nobody is watching when it runs.
set -e

cd "$(dirname "$0")/.."

PROBE_PID="$1"
DECISION_LOG="./output_dir/auto_decision.log"
PERF_DIR="./performance"

BASELINE_JSON="${PERF_DIR}/MuReD_retexpert_seed42_200ep_test_performance.json"
PROBE_JSON="${PERF_DIR}/MuReD_retexpert_seed42_test_performance.json"

echo "[$(date)] Orchestrator started, watching probe PID ${PROBE_PID}" >> "${DECISION_LOG}"

# Poll (not a bash 'wait', since this script did not fork PROBE_PID itself --
# kill -0 works for any process on the machine we have permission to see).
while kill -0 "${PROBE_PID}" 2>/dev/null; do
    sleep 60
done

# Small grace period for the final file writes to land on disk.
sleep 10

echo "[$(date)] Probe process ${PROBE_PID} has exited. Reading results..." >> "${DECISION_LOG}"

if [[ ! -f "${BASELINE_JSON}" || ! -f "${PROBE_JSON}" ]]; then
    echo "[$(date)] ERROR: expected result file missing (baseline=${BASELINE_JSON}, probe=${PROBE_JSON}). Aborting, NOT launching further training." >> "${DECISION_LOG}"
    exit 1
fi

read -r F1_200 KAPPA_200 F1_400 KAPPA_400 <<< "$(.venv/bin/python3 - "${BASELINE_JSON}" "${PROBE_JSON}" <<'EOF'
import json, sys
b = json.load(open(sys.argv[1]))["test_metrics"]
p = json.load(open(sys.argv[2]))["test_metrics"]
print(b["f1"], b["kappa"], p["f1"], p["kappa"])
EOF
)"

DELTA_F1=$(.venv/bin/python3 -c "print(${F1_400} - ${F1_200})")
DELTA_KAPPA=$(.venv/bin/python3 -c "print(${KAPPA_400} - ${KAPPA_200})")

echo "[$(date)] 200-epoch: F1=${F1_200} Kappa=${KAPPA_200}" >> "${DECISION_LOG}"
echo "[$(date)] 400-epoch: F1=${F1_400} Kappa=${KAPPA_400}" >> "${DECISION_LOG}"
echo "[$(date)] Delta: F1=${DELTA_F1} Kappa=${DELTA_KAPPA}" >> "${DECISION_LOG}"

CONFIRMED=$(.venv/bin/python3 -c "print(1 if (${DELTA_F1} >= 0.01 and ${DELTA_KAPPA} >= 0.01) else 0)")

if [[ "${CONFIRMED}" == "1" ]]; then
    echo "[$(date)] HYPOTHESIS CONFIRMED (both F1 and Kappa improved by >= 0.01). Launching remaining 4 seeds at 400 epochs." >> "${DECISION_LOG}"
    nohup ./scripts/run_mured_4seeds_400ep.sh > ./output_dir/mured_4seeds_400ep_driver.log 2>&1 &
    echo "[$(date)] Launched run_mured_4seeds_400ep.sh with driver PID $!" >> "${DECISION_LOG}"
else
    echo "[$(date)] HYPOTHESIS NOT CONFIRMED (improvement below threshold, or a metric regressed). NOT launching further training. Longer training does not appear to close the gap; the data-split/protocol hypothesis is more likely responsible." >> "${DECISION_LOG}"
fi

echo "[$(date)] Orchestrator finished." >> "${DECISION_LOG}"
