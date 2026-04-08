# Scripts

Automation scripts for dataset acquisition, benchmarking, and results visualization.

## Script Reference

| Script | Interpreter | Purpose | Output |
|--------|-------------|---------|--------|
| `download_mnist.sh` | Bash | Download and extract the MNIST dataset (Linux/macOS) | Files in `data/mnist/raw/` |
| `download_mnist.ps1` | PowerShell | Download and extract the MNIST dataset (Windows) | Files in `data/mnist/raw/` |
| `run_training.sh` | Bash | Run a single training configuration N times (Sequential, OpenMP, or MPI). Controlled via `NUM_RUNS` env var (default: 1) | Timestamped CSV in `results/tables/`, per-run logs in `results/logs/` |
| `benchmark.sh` | Bash | Run the full benchmark suite across all network configurations and parallelism levels (1–8 threads/processes). Supports repeated runs via `NUM_RUNS` env var (default: 1) | Timestamped CSV in `results/tables/` |
| `test_benchmark.sh` | Bash | Run a quick 1-epoch validation benchmark (sequential + OpenMP 2 threads + MPI 2 processes). Supports repeated runs via `NUM_RUNS` env var (default: 1) | `results/tables/benchmark_test.csv` |
| `plot_results.py` | Python 3 | Generate ASCII summary tables and PNG speedup bar charts from the most recent benchmark CSV. Aggregates multiple runs (mean ± stddev, error bars) when present | Prints tables to terminal; saves PNGs to `results/plots/` |

## Dependencies

- **`plot_results.py`** requires Python 3. PNG chart generation additionally requires `matplotlib` (`pip install matplotlib`). If matplotlib is not installed, the script still prints ASCII tables to the terminal.
- **`benchmark.sh`** and **`test_benchmark.sh`** require all three training binaries to be built first (`make all`). On Windows, run these scripts through Git Bash.
