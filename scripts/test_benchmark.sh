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

echo "config,implementation,parallelism,run,epoch_time_sec" > "$CSV"

for run in $(seq 1 "$NUM_RUNS"); do
    echo "Testing sequential (run $run/$NUM_RUNS)..."
    ./build/train_seq --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/seq_${cfg_name}_run${run}.log" > "$TMP" 2>&1
    epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$cfg_name,sequential,1,$run,$epoch_time" >> "$CSV"
    echo "  Sequential: $epoch_time s"

    echo "Testing OpenMP (2 threads, run $run/$NUM_RUNS)..."
    ./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads 2 --log "$LOG_DIR/omp_${cfg_name}_2t_run${run}.log" > "$TMP" 2>&1
    epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$cfg_name,openmp,2,$run,$epoch_time" >> "$CSV"
    echo "  OpenMP 2: $epoch_time s"

    echo "Testing MPI (2 procs, run $run/$NUM_RUNS)..."
    mpiexec -np 2 ./build/train_mpi --config "$cfg" --data "$DATA_DIR" --log "$LOG_DIR/mpi_${cfg_name}_2p_run${run}.log" > "$TMP" 2>&1
    epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
    echo "$cfg_name,mpi,2,$run,$epoch_time" >> "$CSV"
    echo "  MPI 2: $epoch_time s"
done

rm -f "$TMP"

echo ""
echo "=== CSV Content ==="
cat "$CSV"
echo ""
echo "Logs saved to $LOG_DIR/"
echo "Test PASSED"
