#!/usr/bin/env bash
# Quick test of the benchmark flow with 1-epoch config.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="data/mnist/raw"
CSV="results/tables/benchmark_test.csv"
cfg="configs/test_benchmark.conf"
cfg_name="test_benchmark"
TMP=$(mktemp)

mkdir -p results/tables
echo "config,implementation,parallelism,epoch_time_sec" > "$CSV"

echo "Testing sequential..."
./build/train_seq --config "$cfg" --data "$DATA_DIR" > "$TMP" 2>&1
epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
echo "$cfg_name,sequential,1,$epoch_time" >> "$CSV"
echo "  Sequential: $epoch_time s"

echo "Testing OpenMP (2 threads)..."
./build/train_omp --config "$cfg" --data "$DATA_DIR" --threads 2 > "$TMP" 2>&1
epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
echo "$cfg_name,openmp,2,$epoch_time" >> "$CSV"
echo "  OpenMP 2: $epoch_time s"

echo "Testing MPI (2 procs)..."
mpiexec -np 2 ./build/train_mpi --config "$cfg" --data "$DATA_DIR" > "$TMP" 2>&1
epoch_time=$(grep "Epoch" "$TMP" | tail -1 | grep -oP 'Time:\s+\K[0-9.]+')
echo "$cfg_name,mpi,2,$epoch_time" >> "$CSV"
echo "  MPI 2: $epoch_time s"

rm -f "$TMP"

echo ""
echo "=== CSV Content ==="
cat "$CSV"
echo ""
echo "Test PASSED"
