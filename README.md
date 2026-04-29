# Parallel MLP Training on MNIST in C/C++ with OpenMP and MPI

A C++ implementation of a Multi-Layer Perceptron (MLP) trained on the MNIST handwritten digit dataset, featuring three training strategies: **sequential** (baseline), **OpenMP** (shared-memory parallelism), and **MPI** (distributed parallelism). The project measures and compares wall-clock training times to evaluate parallel speedup across different network architectures and parallelism levels.

---

## Table of Contents

- [Parallel MLP Training on MNIST in C/C++ with OpenMP and MPI](#parallel-mlp-training-on-mnist-in-cc-with-openmp-and-mpi)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Alternative: Docker (any OS)](#alternative-docker-any-os)
  - [Project Structure](#project-structure)
  - [Network Configurations](#network-configurations)
  - [Step-by-Step Toolchain](#step-by-step-toolchain)
    - [Step 1: Download the MNIST Dataset](#step-1-download-the-mnist-dataset)
    - [Step 2: Build the Project](#step-2-build-the-project)
    - [Step 3: Run Unit Tests](#step-3-run-unit-tests)
    - [Step 4: Train — Sequential Baseline](#step-4-train--sequential-baseline)
    - [Step 5: Train — OpenMP Parallelization](#step-5-train--openmp-parallelization)
    - [Step 6: Train — MPI Distribution](#step-6-train--mpi-distribution)
    - [Step 7: Run Benchmarks](#step-7-run-benchmarks)
      - [Full Benchmark Suite](#full-benchmark-suite)
      - [Quick Validation Benchmark](#quick-validation-benchmark)
    - [Step 8: Visualize Results](#step-8-visualize-results)
  - [VS Code Tasks Reference](#vs-code-tasks-reference)
  - [Docker Setup (Quick Start)](#docker-setup-quick-start)
    - [1. Build the Image](#1-build-the-image)
    - [2. Download MNIST (one-time)](#2-download-mnist-one-time)
    - [3. Train](#3-train)
    - [4. Benchmark](#4-benchmark)
    - [5. Customize](#5-customize)

---

## Prerequisites

Before building and running the project, make sure the following tools are installed and available on your `PATH`:

| Tool | Purpose | Installation Notes |
| ------ | --------- | -------------------- |
| **g++ (MinGW-w64)** | C++17 compiler | Install via [MSYS2](https://www.msys2.org/) or [MinGW-w64](https://www.mingw-w64.org/) |
| **make** | Build automation | Bundled with MSYS2/MinGW; or install via `choco install make` |
| **OpenMP** | Shared-memory threading | Included with g++ (`-fopenmp` flag) |
| **MS-MPI** | Distributed message passing | Install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (both SDK and Runtime) |
| **mpicxx / mpiexec** | MPI compiler wrapper and launcher | Provided by MS-MPI installation |
| **Python 3** | Results visualization | Install from [python.org](https://www.python.org/) |
| **matplotlib** (optional) | PNG chart generation | `pip install matplotlib` |
| **bash** | Benchmark scripts | Available via [Git for Windows](https://gitforwindows.org/) (Git Bash) |

> **Note**: On Linux, replace MS-MPI with OpenMPI or MPICH, and use `scripts/download_mnist.sh` instead of the PowerShell script.

### Alternative: Docker (any OS)

Instead of installing the toolchain manually, you can use Docker to get a fully configured environment with a single command. See [Docker Setup (Quick Start)](#docker-setup-quick-start) below or the full guide at [`docs/docker.md`](docs/docker.md).

| Tool                                                          | Purpose                                                     |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| [Docker](https://docs.docker.com/get-docker/) 20.10+          | Container runtime                                           |
| [Docker Compose](https://docs.docker.com/compose/install/) v2+ | Service orchestration (bundled with Docker Desktop)         |

---

## Project Structure

```text
parallel-mlp-mnist/
├── Makefile                    # Build system — targets: all, seq, omp, mpi, test, clean
├── README.md                   # This file
├── Dockerfile                  # Multi-stage Docker build (builder + runtime)
├── docker-compose.yml          # Compose services for training, benchmarking, plotting
├── .dockerignore               # Docker build context exclusions
├── .devcontainer/
│   ├── Dockerfile              # Dev Container image (compiler + debugger + runtime)
│   └── devcontainer.json       # VS Code Dev Container configuration
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── .vscode/
│   └── tasks.json              # VS Code task definitions for all toolchain actions
├── configs/                    # Network configuration files
│   ├── small_network.conf      #   784 → 128 → 10
│   ├── medium_network.conf     #   784 → 256 → 128 → 10
│   ├── large_network.conf      #   784 → 512 → 256 → 128 → 10
│   └── test_benchmark.conf     #   784 → 128 → 10 (1 epoch, for quick validation)
├── data/
│   └── mnist/raw/              # MNIST binary files (downloaded in Step 1)
├── docs/
│   ├── project_specification.md
│   ├── architecture.md
│   └── docker.md               # Full Docker setup and usage guide
├── include/                    # C++ header files
│   ├── activations.h           #   ReLU, Softmax
│   ├── dataset.h               #   MNIST IDX format loader
│   ├── loss.h                  #   Cross-entropy loss
│   ├── metrics.h               #   Accuracy computation
│   ├── mlp.h                   #   MLP network structure and operations
│   ├── optimizer.h             #   SGD optimizer
│   ├── timer.h                 #   Wall-clock timing utilities
│   └── utils.h                 #   Config parsing, argument handling, logging
├── results/
│   ├── logs/                   # Task logs (build, test, training, benchmarks)
│   ├── plots/                  # Generated PNG speedup charts
│   └── tables/                 # Benchmark CSV files
├── scripts/
│   ├── benchmark.sh            # Full benchmark suite (all configs × all parallelism levels)
│   ├── test_benchmark.sh       # Quick 1-epoch validation benchmark
│   ├── download_mnist.sh       # Dataset download (Linux/macOS)
│   ├── download_mnist.ps1      # Dataset download (Windows PowerShell)
│   └── plot_results.py         # CSV → ASCII tables + PNG bar charts
├── src/                        # C++ source files
│   ├── train_sequential.cpp    #   Sequential training entry point
│   ├── train_openmp.cpp        #   OpenMP training entry point
│   ├── train_mpi.cpp           #   MPI training entry point
│   ├── activations.cpp         #   Activation function implementations
│   ├── dataset.cpp             #   MNIST data loading and normalization
│   ├── loss.cpp                #   Loss computation and gradients
│   ├── metrics.cpp             #   Classification accuracy
│   ├── mlp.cpp                 #   Network creation, forward, backward, update
│   ├── optimizer.cpp           #   SGD weight update
│   ├── timer.cpp               #   High-resolution timing
│   └── utils.cpp               #   Config file parsing, CLI argument parsing
└── tests/                      # Unit tests
    ├── test_dataset.cpp        #   MNIST loading, normalization, label range
    ├── test_loss.cpp           #   ReLU, softmax, cross-entropy, gradient checks
    └── test_mlp.cpp            #   Network lifecycle, forward/backward, loss reduction
```

---

## Network Configurations

Four predefined configurations are provided in `configs/` (small, medium, large, and a quick-test variant) with varying hidden layer depths and parameter counts. See [`configs/README.md`](configs/README.md) for the full configuration file format, key reference, and predefined configuration details.

---

## Step-by-Step Toolchain

This section walks through the complete project workflow from data acquisition to results visualization. Each step can be run independently. Commands are shown for the terminal; the corresponding **VS Code task** name is noted for each step.

### Step 1: Download the MNIST Dataset

**Purpose**: Download the four MNIST binary files (training images, training labels, test images, test labels) into `data/mnist/raw/`.

**Windows (PowerShell)**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1
```

**Linux / macOS**:

```bash
bash scripts/download_mnist.sh
```

**Expected output**: Four IDX binary files in `data/mnist/raw/`. See [`data/README.md`](data/README.md) for file details and format description.

> **VS Code Task**: `Download MNIST Dataset`

---

### Step 2: Build the Project

**Purpose**: Compile the C++ source code into executable binaries. The Makefile provides targets for each training variant independently or all at once.

**Build all variants**:

```bash
make all
```

**Build only a specific variant**:

```bash
make seq    # Sequential binary → build/train_seq
make omp    # OpenMP binary     → build/train_omp
make mpi    # MPI binary        → build/train_mpi
```

**Clean all build artifacts**:

```bash
make clean
```

**Expected output**: Compiled binaries in `build/`:

- `build/train_seq` — Sequential training executable
- `build/train_omp` — OpenMP training executable
- `build/train_mpi` — MPI training executable

> **VS Code Tasks**: `Build All`, `Build Sequential`, `Build OpenMP`, `Build MPI`, `Clean Build`

---

### Step 3: Run Unit Tests

**Purpose**: Verify that core components work correctly before training.

```bash
make test
```

Builds and runs all test suites (dataset loading, activations & loss, MLP forward/backward). See [`tests/README.md`](tests/README.md) for per-file coverage details and how to add new tests.

> **VS Code Task**: `Build & Run Tests`

---

### Step 4: Train — Sequential Baseline

**Purpose**: Train the MLP using the single-threaded sequential implementation. This serves as the baseline for measuring parallel speedup.

```bash
./build/train_seq --config configs/small_network.conf
```

Replace `small_network.conf` with any config file from `configs/`. Optionally specify a custom data directory or log file:

```bash
./build/train_seq --config configs/medium_network.conf --data data/mnist/raw --log results/logs/my_run.log
```

**Expected output** (per epoch):

```text
Epoch 1/10  Loss: 0.4532  Accuracy: 89.12%  Time: 2.345s
Epoch 2/10  Loss: 0.3218  Accuracy: 91.45%  Time: 2.301s
...
Total training time: 23.456s
```

> **VS Code Task**: `Train Sequential` — prompts you to select a configuration file and number of runs. Produces a timestamped CSV in `results/tables/` and per-run logs in `results/logs/`.

---

### Step 5: Train — OpenMP Parallelization

**Purpose**: Train the MLP using OpenMP shared-memory parallelism. Multiple threads process different training samples simultaneously within mini-batches.

```bash
./build/train_omp --config configs/small_network.conf --threads 4
```

**Parameters**:

- `--config <path>` — Network configuration file (required)
- `--threads <N>` — Number of OpenMP threads (default: 2)
- `--data <path>` — MNIST data directory (default: `data/mnist/raw`)
- `--log <path>` — Log file path (optional; writes output to both stdout and file)

**Expected output**: Same format as sequential, but with faster per-epoch times as thread count increases.

> **VS Code Task**: `Train OpenMP` — prompts you to select a configuration file, thread count (1–8), and number of runs. Produces a timestamped CSV in `results/tables/` and per-run logs in `results/logs/`.

---

### Step 6: Train — MPI Distribution

**Purpose**: Train the MLP using MPI distributed parallelism across multiple processes. Each process handles a partition of the training data.

```bash
mpiexec -np 4 ./build/train_mpi --config configs/small_network.conf
```

**Parameters**:

- `-np <N>` — Number of MPI processes (mpiexec argument)
- `--config <path>` — Network configuration file (required)
- `--data <path>` — MNIST data directory (default: `data/mnist/raw`)
- `--log <path>` — Log file path (optional; rank 0 writes output to both stdout and file)

**Expected output**: Same format as sequential (only rank 0 prints output), with faster training as process count increases.

> **Note**: Each mini-batch is split evenly across MPI ranks. Ensure `batch_size` (in the config) is at least as large as the number of processes (`-np`) so every rank receives at least one sample per batch.

> **VS Code Task**: `Train MPI` — prompts you to select a configuration file, process count (1–8), and number of runs. Produces a timestamped CSV in `results/tables/` and per-run logs in `results/logs/`.

---

### Step 7: Run Benchmarks

**Purpose**: Systematically measure training performance across all configurations and parallelism levels, producing a CSV file for analysis.

#### Full Benchmark Suite

Runs all three network configurations (small, medium, large) with:

- Sequential: 1 run
- OpenMP: 1–8 threads
- MPI: 1–8 processes

```bash
bash scripts/benchmark.sh                    # single run (default)
NUM_RUNS=5 bash scripts/benchmark.sh         # 5 repeated runs per configuration
```

**Output**: Timestamped CSV in `results/tables/` and per-run training logs in `results/logs/benchmark_<timestamp>/`. See [`results/README.md`](results/README.md) for the CSV schema, naming conventions, and log structure.

#### Quick Validation Benchmark

Runs a fast 1-epoch test to verify the benchmark infrastructure:

```bash
bash scripts/test_benchmark.sh               # single run (default)
NUM_RUNS=3 bash scripts/test_benchmark.sh    # 3 repeated runs
```

**Output**: `results/tables/benchmark_test.csv` with results for sequential, OpenMP (2 threads), and MPI (2 processes). Per-run logs in `results/logs/test_benchmark_<timestamp>/`.

> **VS Code Tasks**: `Run Full Benchmark`, `Run Quick Test Benchmark`
>
> **Note**: Both benchmark tasks automatically build all binaries first (`Build All` runs as a dependency).

---

### Step 8: Visualize Results

**Purpose**: Process the benchmark CSV to generate human-readable summary tables and speedup charts.

```bash
python scripts/plot_results.py
```

Prints ASCII summary tables to the terminal and generates PNG speedup bar charts (if matplotlib is installed) into `results/plots/`. Output is also logged to `results/logs/plot_results.log`. When the CSV contains multiple runs, summary tables display mean ± stddev with run counts, and speedup charts include error bars. See [`results/README.md`](results/README.md) for output file naming conventions.

> **VS Code Task**: `Plot Results`

---

## VS Code Tasks Reference

All toolchain actions are available as VS Code tasks. Run them via **Terminal → Run Task...** or the keyboard shortcut `Ctrl+Shift+P` → "Tasks: Run Task".

| Task Label | Category | Description | Inputs | Log Output |
| ------------ | ---------- | ------------- | -------- | ------------ |
| **Download MNIST Dataset** | Data | Download and extract MNIST dataset into `data/mnist/raw/` | — | `results/logs/download_mnist.log` |
| **Build All** | Build | Compile all three training binaries (default build task) | — | `results/logs/build_all.log` |
| **Build Sequential** | Build | Compile `build/train_seq` only | — | `results/logs/build_seq.log` |
| **Build OpenMP** | Build | Compile `build/train_omp` only | — | `results/logs/build_omp.log` |
| **Build MPI** | Build | Compile `build/train_mpi` only | — | `results/logs/build_mpi.log` |
| **Clean Build** | Build | Remove all compiled binaries and object files | — | `results/logs/clean_build.log` |
| **Build & Run Tests** | Test | Compile and execute all unit tests (default test task) | — | `results/logs/test_*.log` |
| **Train Sequential** | Train | Train MLP with sequential implementation (auto-builds first) | Config file, Number of runs | `results/logs/train_seq_<config>_<ts>/` |
| **Train OpenMP** | Train | Train MLP with OpenMP parallelism (auto-builds first) | Config file, Thread count, Number of runs | `results/logs/train_omp_<config>_<ts>/` |
| **Train MPI** | Train | Train MLP with MPI distribution (auto-builds first) | Config file, Process count, Number of runs | `results/logs/train_mpi_<config>_<ts>/` |
| **Run Full Benchmark** | Benchmark | Run complete benchmark suite across all configs and parallelism levels with N repeated runs (auto-builds first) | Number of runs | `results/logs/benchmark_<ts>/` |
| **Run Quick Test Benchmark** | Benchmark | Run 1-epoch validation benchmark (auto-builds first) | — | `results/logs/test_benchmark_<ts>/` |
| **Plot Results** | Analysis | Generate summary tables and speedup charts from benchmark CSV | — | `results/logs/plot_results.log` |

**Input prompts**: When running Train or Benchmark tasks, VS Code will prompt you to select:

- **Configuration** (Train tasks): `small_network`, `medium_network`, `large_network`, or `test_benchmark`
- **Thread count** (OpenMP only): 1–8
- **Process count** (MPI only): 1–8
- **Number of runs** (all Train + Full Benchmark tasks): How many times to repeat the training (default: 1)

---

## Docker Setup (Quick Start)

Docker bundles all dependencies (g++, OpenMPI, Python, matplotlib) into a single image, so you don't need to install anything except Docker itself.

### 1. Build the Image

```bash
docker build -t parallel-mlp .
```

### 2. Download MNIST (one-time)

```bash
docker compose run download-data
```

### 3. Train

```bash
# Sequential
docker compose run train-seq

# OpenMP (4 threads)
THREADS=4 docker compose run train-omp

# MPI (4 processes)
PROCS=4 docker compose run train-mpi
```

### 4. Benchmark

```bash
docker compose run benchmark                    # single run
NUM_RUNS=5 docker compose run benchmark          # 5 repeated runs
```

### 5. Customize

Use environment variables to change the configuration, thread/process count, and number of runs:

```bash
CONFIG=configs/large_network.conf THREADS=8 docker compose run train-omp
```

Results are written to `results/` on the host via a volume mount.

> For the complete Docker guide — including direct `docker run` usage, Dev Container setup, CI pipeline details, and troubleshooting — see [`docs/docker.md`](docs/docker.md).
