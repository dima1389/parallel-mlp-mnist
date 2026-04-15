#!/usr/bin/env bash
# Benchmark sequential vs OpenMP vs MPI training across network sizes.
# Runs each configuration and records comprehensive training metrics to a CSV.
#
# Output capture uses temp-file redirection instead of $() subshell pipes
# because MS-MPI mpiexec on Windows does not close pipe FDs properly, causing
# subshell captures to hang.
set -euo pipefail

DATA_DIR="data/mnist/raw"
RESULTS_DIR="results/tables"
CONFIGS=("configs/small_network.conf" "configs/medium_network.conf" "configs/large_network.conf")
OMP_THREADS=(1 2 3 4 5 6 7 8)
MPI_PROCS=(1 2 3 4 5 6 7 8)
NUM_RUNS="${NUM_RUNS:-1}"

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# --- Logging setup ---
LOG_DIR="results/logs/benchmark_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/master.log") 2>&1

# Helper: extract a key's value from the [SUMMARY] block in a file
extract_summary() {
    local file="$1" key="$2"
    sed -n '/\[SUMMARY\]/,/\[\/SUMMARY\]/p' "$file" | grep -oP "^${key}=\K.*" || echo ""
}

# Helper: build a CSV row from the [SUMMARY] block in $TMP
append_summary_row() {
    local cfg_name="$1" impl="$2" par="$3" run="$4" cfg_file="$5"
    local cfg_epochs cfg_batch_size cfg_lr
    cfg_epochs=$(grep -oP '^\s*epochs\s*=\s*\K[0-9]+' "$cfg_file" || echo "0")
    cfg_batch_size=$(grep -oP '^\s*batch_size\s*=\s*\K[0-9]+' "$cfg_file" || echo "0")
    cfg_lr=$(grep -oP '^\s*learning_rate\s*=\s*\K[0-9.]+' "$cfg_file" || echo "0")

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

    echo "$cfg_name,$impl,$par,$run,$cfg_epochs,$cfg_batch_size,$cfg_lr,$total_params,$total_time,$avg_epoch_time,$min_epoch_time,$max_epoch_time,$stddev_epoch_time,$final_loss,$final_accuracy,$best_accuracy,$best_accuracy_epoch,$throughput" >> "$CSV"
}

# CSV header
echo "config,implementation,parallelism,run,epochs,batch_size,learning_rate,total_params,total_time_sec,avg_epoch_time_sec,min_epoch_time_sec,max_epoch_time_sec,stddev_epoch_time_sec,final_loss,final_accuracy,best_accuracy,best_accuracy_epoch,throughput_samples_per_sec" > "$CSV"

for cfg in "${CONFIGS[@]}"; do
    cfg_name=$(basename "$cfg" .conf)
    echo "=== Benchmarking $cfg_name ($NUM_RUNS run(s)) ==="

    for run in $(seq 1 "$NUM_RUNS"); do
        # --- Sequential baseline ---
        echo "  Sequential (run $run/$NUM_RUNS)..."
        ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}_run${run}.log" --epoch-csv "$LOG_DIR/seq_${cfg_name}_run${run}_epochs.csv" > "$TMP" 2>&1
        append_summary_row "$cfg_name" "sequential" "1" "$run" "$cfg"

        # --- OpenMP with varying thread counts ---
        for t in "${OMP_THREADS[@]}"; do
            echo "  OpenMP ($t threads, run $run/$NUM_RUNS)..."
            ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads "$t" --log "$LOG_DIR/omp_${cfg_name}_${t}t_run${run}.log" --epoch-csv "$LOG_DIR/omp_${cfg_name}_${t}t_run${run}_epochs.csv" > "$TMP" 2>&1
            append_summary_row "$cfg_name" "openmp" "$t" "$run" "$cfg"
        done

        # --- MPI with varying process counts ---
        for np in "${MPI_PROCS[@]}"; do
            echo "  MPI ($np procs, run $run/$NUM_RUNS)..."
            mpiexec -np "$np" ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_${np}p_run${run}.log" --epoch-csv "$LOG_DIR/mpi_${cfg_name}_${np}p_run${run}_epochs.csv" > "$TMP" 2>&1
            append_summary_row "$cfg_name" "mpi" "$np" "$run" "$cfg"
        done
    done
done

echo ""
echo "Results saved to $CSV"
echo "Logs saved to $LOG_DIR/"
