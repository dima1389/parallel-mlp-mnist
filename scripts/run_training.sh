#!/usr/bin/env bash
# Run a single training configuration N times, recording epoch times to a CSV.
#
# Usage:
#   NUM_RUNS=<N> bash scripts/run_training.sh <impl> <config> [parallelism]
#
#   impl:        seq | omp | mpi
#   config:      config base name (e.g., small_network)
#   parallelism: thread count (omp) or process count (mpi); ignored for seq
#
# Environment:
#   NUM_RUNS  — number of repeated runs (default: 1)
#
# Output capture uses temp-file redirection instead of $() subshell pipes
# because MS-MPI mpiexec on Windows does not close pipe FDs properly, causing
# subshell captures to hang.
set -euo pipefail

IMPL="${1:?Usage: run_training.sh <seq|omp|mpi> <config> [parallelism]}"
CONFIG="${2:?Usage: run_training.sh <seq|omp|mpi> <config> [parallelism]}"
PARALLELISM="${3:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
DATA_DIR="data/mnist/raw"

# Derive implementation label and parallelism value for CSV
case "$IMPL" in
    seq)
        IMPL_NAME="sequential"
        PAR=1
        ;;
    omp)
        IMPL_NAME="openmp"
        PAR="$PARALLELISM"
        ;;
    mpi)
        IMPL_NAME="mpi"
        PAR="$PARALLELISM"
        ;;
    *)
        echo "Error: unknown implementation '$IMPL'. Use seq, omp, or mpi." >&2
        exit 1
        ;;
esac

# --- Output setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/tables"
mkdir -p "$RESULTS_DIR"

CSV="$RESULTS_DIR/train_${IMPL}_${CONFIG}_${TIMESTAMP}.csv"
LOG_DIR="results/logs/train_${IMPL}_${CONFIG}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "config,implementation,parallelism,run,epoch_time_sec" > "$CSV"

echo "=== Training $CONFIG ($IMPL_NAME, parallelism=$PAR, $NUM_RUNS run(s)) ==="

for run in $(seq 1 "$NUM_RUNS"); do
    LOG_FILE="$LOG_DIR/run${run}.log"
    echo "  Run $run/$NUM_RUNS..."

    case "$IMPL" in
        seq)
            ./build/train_seq --config "configs/${CONFIG}.conf" --data "$DATA_DIR" --log "$LOG_FILE" > "$TMP" 2>&1
            ;;
        omp)
            ./build/train_omp --config "configs/${CONFIG}.conf" --data "$DATA_DIR" --threads "$PAR" --log "$LOG_FILE" > "$TMP" 2>&1
            ;;
        mpi)
            mpiexec -np "$PAR" ./build/train_mpi --config "configs/${CONFIG}.conf" --data "$DATA_DIR" --log "$LOG_FILE" > "$TMP" 2>&1
            ;;
    esac

    epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$CONFIG,$IMPL_NAME,$PAR,$run,$epoch_time" >> "$CSV"
    echo "    Epoch time: $epoch_time s"
done

echo ""
echo "Results saved to $CSV"
echo "Logs saved to $LOG_DIR/"
