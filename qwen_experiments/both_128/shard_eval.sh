#!/usr/bin/env bash
set -euo pipefail

########################
# CONFIG
########################

EXP_NAME=both_128

# Number of shards / parallel processes
NUM_SHARDS=3

# Paths (edit these to your actual locations)
JSON_PATH="/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/test_vqa.json"
VIDEO_DIR="/home/aryan/ami/381V-final/data/gaze_crops_128"
CSV_PATH="/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/P02_videos.csv"

# Base output dirs
LOG_DIR="/home/aryan/ami/381V-final/381V-final-project/qwen_experiments_clean/${EXP_NAME}/logs"
RESULTS_DIR="/home/aryan/ami/381V-final/381V-final-project/qwen_experiments_clean/${EXP_NAME}/results"

# Hyperparameters (shared across shards)
NUM_FRAMES=8
SAMPLE_FPS=0.25
TOTAL_PIXELS=$((8 * 256 * 256))
MAX_NEW_TOKENS=8
MAX_PER_TYPE=100

# Your eval script
PY_SCRIPT="/home/aryan/ami/381V-final/381V-final-project/Qwen3-VL/custom/video_inference.py"

########################
# SETUP
########################

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "Launching ${NUM_SHARDS} shards..."
echo "Logs   -> ${LOG_DIR}"
echo "CSVs   -> ${RESULTS_DIR}"
echo

########################
# LAUNCH SHARDS
########################

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    LOGFILE="${LOG_DIR}/shard_${SHARD_ID}.log"
    RESULTS_CSV="${RESULTS_DIR}/qwen3_vl_results_shard_${SHARD_ID}.csv"

    echo "Starting shard ${SHARD_ID}..."
    echo "  Logfile:     ${LOGFILE}"
    echo "  Results CSV: ${RESULTS_CSV}"

    # If you have multiple GPUs, you can pin per shard like:
    # export CUDA_VISIBLE_DEVICES=${SHARD_ID}

    python "${PY_SCRIPT}" \
        --json-path "${JSON_PATH}" \
        --video-dir "${VIDEO_DIR}" \
        --csv-path "${CSV_PATH}" \
        --results-csv "${RESULTS_CSV}" \
        --num-frames "${NUM_FRAMES}" \
        --sample-fps "${SAMPLE_FPS}" \
        --total-pixels "${TOTAL_PIXELS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-per-type "${MAX_PER_TYPE}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-id "${SHARD_ID}" \
        > "${LOGFILE}" 2>&1 &
done

echo
echo "All shards launched. Waiting for them to finish..."
wait
echo "All shards finished."
echo "Done."
