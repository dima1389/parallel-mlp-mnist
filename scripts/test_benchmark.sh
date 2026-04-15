#!/usr/bin/env bash
# Quick test of the benchmark flow with 1-epoch config.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="data/mnist/raw"
cfg="configs/test_benchmark.conf"
cfg_name="test_benchmark"
NUM_RUNS="${NUM_RUNS:-1}"
TMP=$(mktemp)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/test_benchmark_${TIMESTAMP}"
LOG_DIR="$RUN_DIR/logs"
TABLE_DIR="$RUN_DIR/tables"
EPOCH_DIR="$TABLE_DIR/epochs"
PLOT_DIR="$RUN_DIR/plots"
CSV="$TABLE_DIR/summary.csv"
ALL_EPOCHS="$TABLE_DIR/all_epochs.csv"

mkdir -p "$LOG_DIR" "$EPOCH_DIR" "$PLOT_DIR"

exec > >(tee "$LOG_DIR/master.log") 2>&1

# Read hyperparameters from config file
CFG_EPOCHS=$(grep -oP '^\s*epochs\s*=\s*\K[0-9]+' "$cfg" || echo "0")
CFG_BATCH_SIZE=$(grep -oP '^\s*batch_size\s*=\s*\K[0-9]+' "$cfg" || echo "0")
CFG_LR=$(grep -oP '^\s*learning_rate\s*=\s*\K[0-9.]+' "$cfg" || echo "0")

# Helper: extract a key's value from the [SUMMARY] block in a file
extract_summary() {
    local file="$1" key="$2"
    sed -n '/\[SUMMARY\]/,/\[\/SUMMARY\]/p' "$file" | grep -oP "^${key}=\K.*" || echo ""
}

# Helper: build and append a CSV row from the [SUMMARY] block in $TMP
append_summary_row() {
    local impl="$1" par="$2" run="$3"
    local total_time avg_epoch_time min_epoch_time max_epoch_time stddev_epoch_time
    local final_loss final_accuracy best_accuracy best_accuracy_epoch throughput total_params
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
    echo "$cfg_name,$impl,$par,$run,$CFG_EPOCHS,$CFG_BATCH_SIZE,$CFG_LR,$total_params,$total_time,$avg_epoch_time,$min_epoch_time,$max_epoch_time,$stddev_epoch_time,$final_loss,$final_accuracy,$best_accuracy,$best_accuracy_epoch,$throughput" >> "$CSV"
}

echo "config,implementation,parallelism,run,epochs,batch_size,learning_rate,total_params,total_time_sec,avg_epoch_time_sec,min_epoch_time_sec,max_epoch_time_sec,stddev_epoch_time_sec,final_loss,final_accuracy,best_accuracy,best_accuracy_epoch,throughput_samples_per_sec" > "$CSV"

# Consolidated epoch CSV header
echo "config,implementation,parallelism,run,epoch,train_loss,test_accuracy,epoch_time_sec,cumulative_time_sec" > "$ALL_EPOCHS"

# Helper: append epoch data with metadata columns to all_epochs.csv
append_epoch_data() {
    local epoch_csv="$1" impl="$2" par="$3" run="$4"
    if [ -f "$epoch_csv" ]; then
        tail -n +2 "$epoch_csv" | awk -v c="$cfg_name" -v i="$impl" -v p="$par" -v r="$run" \
            -F',' '{print c","i","p","r","$0}' >> "$ALL_EPOCHS"
    fi
}

for run in $(seq 1 "$NUM_RUNS"); do
    echo "Testing sequential (run $run/$NUM_RUNS)..."
    EPOCH_CSV_FILE="$EPOCH_DIR/seq_${cfg_name}_run${run}_epochs.csv"
    ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
    append_summary_row "sequential" "1" "$run"
    append_epoch_data "$EPOCH_CSV_FILE" "sequential" "1" "$run"
    echo "  Sequential: $(extract_summary "$TMP" "total_time") s"

    echo "Testing OpenMP (2 threads, run $run/$NUM_RUNS)..."
    EPOCH_CSV_FILE="$EPOCH_DIR/omp_${cfg_name}_2t_run${run}_epochs.csv"
    ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads 2 --log "$LOG_DIR/omp_${cfg_name}_2t_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
    append_summary_row "openmp" "2" "$run"
    append_epoch_data "$EPOCH_CSV_FILE" "openmp" "2" "$run"
    echo "  OpenMP 2: $(extract_summary "$TMP" "total_time") s"

    echo "Testing MPI (2 procs, run $run/$NUM_RUNS)..."
    EPOCH_CSV_FILE="$EPOCH_DIR/mpi_${cfg_name}_2p_run${run}_epochs.csv"
    mpiexec -np 2 ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_2p_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
    append_summary_row "mpi" "2" "$run"
    append_epoch_data "$EPOCH_CSV_FILE" "mpi" "2" "$run"
    echo "  MPI 2: $(extract_summary "$TMP" "total_time") s"
done

rm -f "$TMP"

echo ""
echo "=== CSV Content ==="
cat "$CSV"
echo ""
echo "Results saved to $RUN_DIR/"
echo "Test PASSED"

# --- Generate plots ---
if command -v python &>/dev/null; then
    echo "Generating plots..."
    python scripts/plot_results.py "$RUN_DIR"
else
    echo "Python not found — skipping plot generation."
fi
