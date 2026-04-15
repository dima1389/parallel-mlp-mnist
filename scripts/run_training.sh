#!/usr/bin/env bash
# Run a single training configuration N times, recording metrics to a CSV.
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

# Read hyperparameters from config file for CSV row
CFG_FILE="configs/${CONFIG}.conf"
CFG_EPOCHS=$(grep -oP '^\s*epochs\s*=\s*\K[0-9]+' "$CFG_FILE" || echo "0")
CFG_BATCH_SIZE=$(grep -oP '^\s*batch_size\s*=\s*\K[0-9]+' "$CFG_FILE" || echo "0")
CFG_LR=$(grep -oP '^\s*learning_rate\s*=\s*\K[0-9.]+' "$CFG_FILE" || echo "0")

# Helper: extract a key's value from the [SUMMARY] block in a file
extract_summary() {
    local file="$1" key="$2"
    sed -n '/\[SUMMARY\]/,/\[\/SUMMARY\]/p' "$file" | grep -oP "^${key}=\K.*" || echo ""
}

# --- Output setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/train_${IMPL}_${CONFIG}_${TIMESTAMP}"
LOG_DIR="$RUN_DIR/logs"
TABLE_DIR="$RUN_DIR/tables"
EPOCH_DIR="$TABLE_DIR/epochs"
PLOT_DIR="$RUN_DIR/plots"
CSV="$TABLE_DIR/summary.csv"
ALL_EPOCHS="$TABLE_DIR/all_epochs.csv"

mkdir -p "$LOG_DIR" "$EPOCH_DIR" "$PLOT_DIR"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "config,implementation,parallelism,run,epochs,batch_size,learning_rate,total_params,total_time_sec,avg_epoch_time_sec,min_epoch_time_sec,max_epoch_time_sec,stddev_epoch_time_sec,final_loss,final_accuracy,best_accuracy,best_accuracy_epoch,throughput_samples_per_sec" > "$CSV"

# Consolidated epoch CSV header
echo "config,implementation,parallelism,run,epoch,train_loss,test_accuracy,epoch_time_sec,cumulative_time_sec" > "$ALL_EPOCHS"

echo "=== Training $CONFIG ($IMPL_NAME, parallelism=$PAR, $NUM_RUNS run(s)) ==="

for run in $(seq 1 "$NUM_RUNS"); do
    LOG_FILE="$LOG_DIR/run${run}.log"
    EPOCH_CSV="$EPOCH_DIR/run${run}_epochs.csv"
    echo "  Run $run/$NUM_RUNS..."

    case "$IMPL" in
        seq)
            ./build/train_seq --config "$CFG_FILE" --data "$DATA_DIR" --log "$LOG_FILE" --epoch-csv "$EPOCH_CSV" > "$TMP" 2>&1
            ;;
        omp)
            ./build/train_omp --config "$CFG_FILE" --data "$DATA_DIR" --threads "$PAR" --log "$LOG_FILE" --epoch-csv "$EPOCH_CSV" > "$TMP" 2>&1
            ;;
        mpi)
            mpiexec -np "$PAR" ./build/train_mpi --config "$CFG_FILE" --data "$DATA_DIR" --log "$LOG_FILE" --epoch-csv "$EPOCH_CSV" > "$TMP" 2>&1
            ;;
    esac

    # Parse [SUMMARY] block from training output
    total_time=$(extract_summary "$TMP" "total_time")
    avg_epoch_time=$(extract_summary "$TMP" "avg_epoch_time")
    min_epoch_time=$(extract_summary "$TMP" "min_epoch_time")
    max_epoch_time=$(extract_summary "$TMP" "max_epoch_time")
    stddev_epoch_time=$(extract_summary "$TMP" "stddev_epoch_time")
    final_loss=$(extract_summary "$TMP" "final_loss")
    final_accuracy=$(extract_summary "$TMP" "final_accuracy")
    best_accuracy=$(extract_summary "$TMP" "best_accuracy")
    best_accuracy_epoch=$(extract_summary "$TMP" "best_accuracy_epoch")
    throughput=$(extract_summary "$TMP" "throughput_samples_per_sec")
    total_params=$(extract_summary "$TMP" "total_params")

    echo "$CONFIG,$IMPL_NAME,$PAR,$run,$CFG_EPOCHS,$CFG_BATCH_SIZE,$CFG_LR,$total_params,$total_time,$avg_epoch_time,$min_epoch_time,$max_epoch_time,$stddev_epoch_time,$final_loss,$final_accuracy,$best_accuracy,$best_accuracy_epoch,$throughput" >> "$CSV"
    echo "    Total time: ${total_time} s  Final accuracy: ${final_accuracy}"

    # Append epoch data with metadata columns to consolidated CSV
    if [ -f "$EPOCH_CSV" ]; then
        tail -n +2 "$EPOCH_CSV" | awk -v c="$CONFIG" -v i="$IMPL_NAME" -v p="$PAR" -v r="$run" \
            -F',' '{print c","i","p","r","$0}' >> "$ALL_EPOCHS"
    fi
done

echo ""
echo "Results saved to $RUN_DIR/"

# --- Generate plots ---
if command -v python &>/dev/null; then
    echo "Generating plots..."
    python scripts/plot_results.py "$RUN_DIR"
else
    echo "Python not found — skipping plot generation."
fi
