#!/usr/bin/env bash
# Quick test of the benchmark flow with 1-epoch config.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="data/mnist/raw"
CSV="results/tables/benchmark_test.csv"
cfg="configs/test_benchmark.conf"
cfg_name="test_benchmark"
NUM_RUNS="${NUM_RUNS:-1}"
TMP=$(mktemp)

mkdir -p results/tables

# --- Logging setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/logs/test_benchmark_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
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

for run in $(seq 1 "$NUM_RUNS"); do
    echo "Testing sequential (run $run/$NUM_RUNS)..."
    ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}_run${run}.log" --epoch-csv "$LOG_DIR/seq_${cfg_name}_run${run}_epochs.csv" > "$TMP" 2>&1
    append_summary_row "sequential" "1" "$run"
    echo "  Sequential: $(extract_summary "$TMP" "total_time") s"

    echo "Testing OpenMP (2 threads, run $run/$NUM_RUNS)..."
    ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads 2 --log "$LOG_DIR/omp_${cfg_name}_2t_run${run}.log" --epoch-csv "$LOG_DIR/omp_${cfg_name}_2t_run${run}_epochs.csv" > "$TMP" 2>&1
    append_summary_row "openmp" "2" "$run"
    echo "  OpenMP 2: $(extract_summary "$TMP" "total_time") s"

    echo "Testing MPI (2 procs, run $run/$NUM_RUNS)..."
    mpiexec -np 2 ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_2p_run${run}.log" --epoch-csv "$LOG_DIR/mpi_${cfg_name}_2p_run${run}_epochs.csv" > "$TMP" 2>&1
    append_summary_row "mpi" "2" "$run"
    echo "  MPI 2: $(extract_summary "$TMP" "total_time") s"
done

rm -f "$TMP"

echo ""
echo "=== CSV Content ==="
cat "$CSV"
echo ""
echo "Logs saved to $LOG_DIR/"
echo "Test PASSED"
