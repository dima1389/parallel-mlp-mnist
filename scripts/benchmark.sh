#!/usr/bin/env bash
# Benchmark sequential vs OpenMP vs MPI training across network sizes.
# Runs each configuration and records the last epoch's wall-clock time to a CSV.
#
# Output capture uses temp-file redirection instead of $() subshell pipes
# because MS-MPI mpiexec on Windows does not close pipe FDs properly, causing
# subshell captures to hang.
set -euo pipefail

DATA_DIR="data/mnist/raw"
RESULTS_DIR="results/tables"
CONFIGS=("configs/small_network.conf" "configs/medium_network.conf" "configs/large_network.conf")
OMP_THREADS=(1 2 4 8)
MPI_PROCS=(1 2 4 8)

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# --- Logging setup ---
LOG_DIR="results/logs/benchmark_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/master.log") 2>&1

# CSV header
echo "config,implementation,parallelism,epoch_time_sec" > "$CSV"

for cfg in "${CONFIGS[@]}"; do
    cfg_name=$(basename "$cfg" .conf)
    echo "=== Benchmarking $cfg_name ==="

    # --- Sequential baseline ---
    echo "  Sequential..."
    ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}.log" > "$TMP" 2>&1
    epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$cfg_name,sequential,1,$epoch_time" >> "$CSV"

    # --- OpenMP with varying thread counts ---
    for t in "${OMP_THREADS[@]}"; do
        echo "  OpenMP ($t threads)..."
        ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads "$t" --log "$LOG_DIR/omp_${cfg_name}_${t}t.log" > "$TMP" 2>&1
        epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
        echo "$cfg_name,openmp,$t,$epoch_time" >> "$CSV"
    done

    # --- MPI with varying process counts ---
    for np in "${MPI_PROCS[@]}"; do
        echo "  MPI ($np procs)..."
        mpiexec -np "$np" ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_${np}p.log" > "$TMP" 2>&1
        epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
        echo "$cfg_name,mpi,$np,$epoch_time" >> "$CSV"
    done
done

echo ""
echo "Results saved to $CSV"
echo "Logs saved to $LOG_DIR/"
