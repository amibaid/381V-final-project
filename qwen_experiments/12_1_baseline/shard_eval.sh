#!/usr/bin/env bash
set -euo pipefail

########################
# CONFIG
########################

EXP_NAME=12_1_baseline

# Number of shards / parallel processes
NUM_SHARDS=3

# Paths (edit these to your actual locations)
JSON_PATH="/content/drive/MyDrive/381V-final-project/HD-EPIC/test_vqa.json"
VIDEO_DIR="/content/drive/MyDrive/381V final project/eval data/trimmed_clips"
CSV_PATH="/content/drive/MyDrive/381V-final-project/HD-EPIC/P02_videos.csv"

# Base output dirs
LOG_DIR="/content/drive/MyDrive/381V-final-project/qwen_experiments/${EXP_NAME}/logs"
RESULTS_DIR="/content/drive/MyDrive/381V-final-project/qwen_experiments/${EXP_NAME}/results"

# Hyperparameters (shared across shards)
NUM_FRAMES=4
SAMPLE_FPS=0.25
TOTAL_PIXELS=$((4 * 256 * 256))
MAX_NEW_TOKENS=8

# Your eval script
PY_SCRIPT="/content/drive/MyDrive/381V-final-project/Qwen3-VL/custom/video_inference.py"

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
        --num-shards "${NUM_SHARDS}" \
        --shard-id "${SHARD_ID}" \
        > "${LOGFILE}" 2>&1 &
done

echo
echo "All shards launched. Waiting for them to finish..."
wait
echo "All shards finished."

echo "Merging shard CSVs into: ${MERGED_CSV}"

shopt -s nullglob
shard_files=("${RESULTS_DIR}/qwen3_vl_results_shard_"*.csv)
shopt -u nullglob

if [ ${#shard_files[@]} -eq 0 ]; then
    echo "No shard CSV files found in ${RESULTS_DIR}. Skipping merge."
else
    # Use the first file's header
    first="${shard_files[0]}"
    echo "Using header from: ${first}"
    cat "${first}" > "${MERGED_CSV}"

    # Append data rows (skip header) from the rest
    for f in "${shard_files[@]:1}"; do
        echo "Appending: ${f}"
        tail -n +2 "${f}" >> "${MERGED_CSV}"
    done

    echo "Merged CSV written to: ${MERGED_CSV}"
fi

echo "Done."
