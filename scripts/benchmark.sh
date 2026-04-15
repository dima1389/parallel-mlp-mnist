#!/usr/bin/env bash
# Benchmark sequential vs OpenMP vs MPI training across network sizes.
# Runs each configuration and records comprehensive training metrics to a CSV.
#
# All outputs (logs, CSVs, plots) are written to a single self-contained
# timestamped directory: results/benchmark_<YYYYMMDD_HHMMSS>/
#
# Output capture uses temp-file redirection instead of $() subshell pipes
# because MS-MPI mpiexec on Windows does not close pipe FDs properly, causing
# subshell captures to hang.
set -euo pipefail

DATA_DIR="data/mnist/raw"
CONFIGS=("configs/small_network.conf" "configs/medium_network.conf" "configs/large_network.conf")
OMP_THREADS=(1 2 3 4 5 6 7 8)
MPI_PROCS=(1 2 3 4 5 6 7 8)
NUM_RUNS="${NUM_RUNS:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/benchmark_${TIMESTAMP}"
LOG_DIR="$RUN_DIR/logs"
TABLE_DIR="$RUN_DIR/tables"
EPOCH_DIR="$TABLE_DIR/epochs"
PLOT_DIR="$RUN_DIR/plots"
CSV="$TABLE_DIR/summary.csv"
ALL_EPOCHS="$TABLE_DIR/all_epochs.csv"

mkdir -p "$LOG_DIR" "$EPOCH_DIR" "$PLOT_DIR"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

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

# Consolidated epoch CSV header
echo "config,implementation,parallelism,run,epoch,train_loss,test_accuracy,epoch_time_sec,cumulative_time_sec" > "$ALL_EPOCHS"

# Helper: append epoch data with metadata columns to all_epochs.csv
append_epoch_data() {
    local epoch_csv="$1" cfg_name="$2" impl="$3" par="$4" run="$5"
    if [ -f "$epoch_csv" ]; then
        tail -n +2 "$epoch_csv" | awk -v c="$cfg_name" -v i="$impl" -v p="$par" -v r="$run" \
            -F',' '{print c","i","p","r","$0}' >> "$ALL_EPOCHS"
    fi
}

for cfg in "${CONFIGS[@]}"; do
    cfg_name=$(basename "$cfg" .conf)
    echo "=== Benchmarking $cfg_name ($NUM_RUNS run(s)) ==="

    for run in $(seq 1 "$NUM_RUNS"); do
        # --- Sequential baseline ---
        echo "  Sequential (run $run/$NUM_RUNS)..."
        EPOCH_CSV_FILE="$EPOCH_DIR/seq_${cfg_name}_run${run}_epochs.csv"
        ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
        append_summary_row "$cfg_name" "sequential" "1" "$run" "$cfg"
        append_epoch_data "$EPOCH_CSV_FILE" "$cfg_name" "sequential" "1" "$run"

        # --- OpenMP with varying thread counts ---
        for t in "${OMP_THREADS[@]}"; do
            echo "  OpenMP ($t threads, run $run/$NUM_RUNS)..."
            EPOCH_CSV_FILE="$EPOCH_DIR/omp_${cfg_name}_${t}t_run${run}_epochs.csv"
            ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads "$t" --log "$LOG_DIR/omp_${cfg_name}_${t}t_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
            append_summary_row "$cfg_name" "openmp" "$t" "$run" "$cfg"
            append_epoch_data "$EPOCH_CSV_FILE" "$cfg_name" "openmp" "$t" "$run"
        done

        # --- MPI with varying process counts ---
        for np in "${MPI_PROCS[@]}"; do
            echo "  MPI ($np procs, run $run/$NUM_RUNS)..."
            EPOCH_CSV_FILE="$EPOCH_DIR/mpi_${cfg_name}_${np}p_run${run}_epochs.csv"
            mpiexec -np "$np" ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_${np}p_run${run}.log" --epoch-csv "$EPOCH_CSV_FILE" > "$TMP" 2>&1
            append_summary_row "$cfg_name" "mpi" "$np" "$run" "$cfg"
            append_epoch_data "$EPOCH_CSV_FILE" "$cfg_name" "mpi" "$np" "$run"
        done
    done
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
