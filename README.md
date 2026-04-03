# Parallel MLP Training on MNIST in C/C++ with OpenMP and MPI

A C++ implementation of a Multi-Layer Perceptron (MLP) trained on the MNIST handwritten digit dataset, featuring three training strategies: **sequential** (baseline), **OpenMP** (shared-memory parallelism), and **MPI** (distributed parallelism). The project measures and compares wall-clock training times to evaluate parallel speedup across different network architectures and parallelism levels.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Network Configurations](#network-configurations)
4. [Step-by-Step Toolchain](#step-by-step-toolchain)
   - [Step 1: Download the MNIST Dataset](#step-1-download-the-mnist-dataset)
   - [Step 2: Build the Project](#step-2-build-the-project)
   - [Step 3: Run Unit Tests](#step-3-run-unit-tests)
   - [Step 4: Train — Sequential Baseline](#step-4-train--sequential-baseline)
   - [Step 5: Train — OpenMP Parallelization](#step-5-train--openmp-parallelization)
   - [Step 6: Train — MPI Distribution](#step-6-train--mpi-distribution)
   - [Step 7: Run Benchmarks](#step-7-run-benchmarks)
   - [Step 8: Visualize Results](#step-8-visualize-results)
5. [VS Code Tasks Reference](#vs-code-tasks-reference)
6. [Configuration File Format](#configuration-file-format)

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

---

## Project Structure

```text
parallel-mlp-mnist/
├── Makefile                    # Build system — targets: all, seq, omp, mpi, test, clean
├── README.md                   # This file
├── configs/                    # Network configuration files
│   ├── small_network.conf      #   784 → 128 → 10
│   ├── medium_network.conf     #   784 → 256 → 128 → 10
│   ├── large_network.conf      #   784 → 512 → 256 → 128 → 10
│   └── test_benchmark.conf     #   784 → 128 → 10 (1 epoch, for quick validation)
├── data/
│   └── mnist/raw/              # MNIST binary files (downloaded in Step 1)
├── docs/
│   ├── project_specification.md
│   └── architecture.md
├── include/                    # C++ header files
│   ├── activations.h           #   ReLU, Softmax
│   ├── dataset.h               #   MNIST IDX format loader
│   ├── loss.h                  #   Cross-entropy loss
│   ├── metrics.h               #   Accuracy computation
│   ├── mlp.h                   #   MLP network structure and operations
│   ├── optimizer.h             #   SGD optimizer
│   ├── timer.h                 #   Wall-clock timing utilities
│   └── utils.h                 #   Config parsing, argument handling
├── results/
│   ├── logs/                   # Training logs
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
├── tests/                      # Unit tests
│   ├── test_dataset.cpp        #   MNIST loading, normalization, label range
│   ├── test_loss.cpp           #   ReLU, softmax, cross-entropy, gradient checks
│   └── test_mlp.cpp            #   Network lifecycle, forward/backward, loss reduction
└── .vscode/
    └── tasks.json              # VS Code task definitions for all toolchain actions
```

---

## Network Configurations

Four predefined configurations are provided in `configs/`:

| Config File | Hidden Layers | Architecture | ~Parameters | Epochs |
| ------------- | --------------- | ------------- | ------------- | -------- |
| `small_network.conf` | 128 | 784 → 128 → 10 | 101,770 | 10 |
| `medium_network.conf` | 256, 128 | 784 → 256 → 128 → 10 | 234,890 | 10 |
| `large_network.conf` | 512, 256, 128 | 784 → 512 → 256 → 128 → 10 | 467,338 | 10 |
| `test_benchmark.conf` | 128 | 784 → 128 → 10 | 101,770 | 1 |

All configurations use: `learning_rate=0.01`, `batch_size=1` (online SGD).

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

**Expected output**: Four files in `data/mnist/raw/`:

- `train-images-idx3-ubyte` — 60,000 training images (28×28 pixels)
- `train-labels-idx1-ubyte` — 60,000 training labels (digits 0–9)
- `t10k-images-idx3-ubyte` — 10,000 test images
- `t10k-labels-idx1-ubyte` — 10,000 test labels

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
make omp    # OpenMP binary    → build/train_omp
make mpi    # MPI binary       → build/train_mpi
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

**Purpose**: Verify that core components (MNIST loading, activation functions, loss computation, MLP forward/backward pass) work correctly before training.

```bash
make test
```

This builds and runs three test suites:

- **test_dataset** — MNIST loading, pixel normalization to [0,1], label range [0–9]
- **test_loss** — ReLU activation, softmax numerical stability, cross-entropy loss, gradient correctness
- **test_mlp** — Network creation/destruction, forward pass output shape, backward pass, loss reduction over 50 steps

**Expected output**: All tests pass with `PASS` messages. Any failure prints detailed error information.

> **VS Code Task**: `Build & Run Tests`

---

### Step 4: Train — Sequential Baseline

**Purpose**: Train the MLP using the single-threaded sequential implementation. This serves as the baseline for measuring parallel speedup.

```bash
./build/train_seq --config configs/small_network.conf
```

Replace `small_network.conf` with any config file from `configs/`. Optionally specify a custom data directory:

```bash
./build/train_seq --config configs/medium_network.conf --data data/mnist/raw
```

**Expected output** (per epoch):

```text
Epoch 1/10  Loss: 0.4532  Accuracy: 89.12%  Time: 2.345s
Epoch 2/10  Loss: 0.3218  Accuracy: 91.45%  Time: 2.301s
...
Total training time: 23.456s
```

> **VS Code Task**: `Train Sequential` — prompts you to select a configuration file.

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

**How it works**: Creates N thread-local network copies. For each mini-batch of N samples, all threads compute forward/backward passes in parallel, then gradients are aggregated and averaged before updating the master network.

**Expected output**: Same format as sequential, but with faster per-epoch times as thread count increases.

> **VS Code Task**: `Train OpenMP` — prompts you to select a configuration file and thread count (1, 2, 4, or 8).

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

**How it works**: The 60,000 training samples are partitioned across N ranks. Each rank computes gradients on its local partition. After each sample, `MPI_Allreduce` synchronizes gradients across all ranks, which are averaged and applied via SGD. All ranks maintain identical network weights.

**Expected output**: Same format as sequential (only rank 0 prints output), with faster training as process count increases.

> **VS Code Task**: `Train MPI` — prompts you to select a configuration file and process count (1, 2, 4, or 8).

---

### Step 7: Run Benchmarks

**Purpose**: Systematically measure training performance across all configurations and parallelism levels, producing a CSV file for analysis.

#### Full Benchmark Suite

Runs all three network configurations (small, medium, large) with:

- Sequential: 1 run
- OpenMP: 1, 2, 4, 8 threads
- MPI: 1, 2, 4, 8 processes

```bash
bash scripts/benchmark.sh
```

**Output**: Timestamped CSV at `results/tables/benchmark_YYYYMMDD_HHMMSS.csv` with columns:

```text
config,implementation,parallelism,epoch_time_sec
```

#### Quick Validation Benchmark

Runs a fast 1-epoch test to verify the benchmark infrastructure:

```bash
bash scripts/test_benchmark.sh
```

**Output**: `results/tables/benchmark_test.csv` with results for sequential, OpenMP (2 threads), and MPI (2 processes).

> **VS Code Tasks**: `Run Full Benchmark`, `Run Quick Test Benchmark`
>
> **Note**: Both benchmark tasks automatically build all binaries first (`Build All` runs as a dependency).

---

### Step 8: Visualize Results

**Purpose**: Process the benchmark CSV to generate human-readable summary tables and speedup charts.

```bash
python scripts/plot_results.py
```

**Outputs**:

1. **ASCII summary table** (printed to terminal) — per configuration, shows each implementation's wall-clock time and speedup relative to the sequential baseline.
2. **PNG bar charts** (if matplotlib is installed) — saved to `results/plots/speedup_<config>.png`, one chart per network configuration showing speedup by implementation and parallelism level.

**Sample terminal output**:

```text
=== small_network ===
Implementation       Time [s]    Speedup
sequential (1)       2.345       1.00x
openmp (2)           1.234       1.90x
openmp (4)           0.678       3.46x
mpi (2)              1.456       1.61x
mpi (4)              0.812       2.89x
...
```

> **VS Code Task**: `Plot Results`

---

## VS Code Tasks Reference

All toolchain actions are available as VS Code tasks. Run them via **Terminal → Run Task...** or the keyboard shortcut `Ctrl+Shift+P` → "Tasks: Run Task".

| Task Label | Category | Description | Inputs |
| ------------ | ---------- | ------------- | -------- |
| **Download MNIST Dataset** | Data | Download and extract MNIST dataset into `data/mnist/raw/` | — |
| **Build All** | Build | Compile all three training binaries (default build task) | — |
| **Build Sequential** | Build | Compile `build/train_seq` only | — |
| **Build OpenMP** | Build | Compile `build/train_omp` only | — |
| **Build MPI** | Build | Compile `build/train_mpi` only | — |
| **Clean Build** | Build | Remove all compiled binaries and object files | — |
| **Build & Run Tests** | Test | Compile and execute all unit tests (default test task) | — |
| **Train Sequential** | Train | Train MLP with sequential implementation (auto-builds first) | Config file |
| **Train OpenMP** | Train | Train MLP with OpenMP parallelism (auto-builds first) | Config file, Thread count |
| **Train MPI** | Train | Train MLP with MPI distribution (auto-builds first) | Config file, Process count |
| **Run Full Benchmark** | Benchmark | Run complete benchmark suite across all configs and parallelism levels (auto-builds first) | — |
| **Run Quick Test Benchmark** | Benchmark | Run 1-epoch validation benchmark (auto-builds first) | — |
| **Plot Results** | Analysis | Generate summary tables and speedup charts from benchmark CSV | — |

**Input prompts**: When running Train tasks, VS Code will prompt you to select:

- **Configuration**: `small_network`, `medium_network`, `large_network`, or `test_benchmark`
- **Thread count** (OpenMP only): 1, 2, 4, or 8
- **Process count** (MPI only): 1, 2, 4, or 8

---

## Configuration File Format

Configuration files use a simple `key=value` format:

```ini
hidden_layers=256,128
epochs=10
learning_rate=0.01
batch_size=1
```

| Key | Type | Description |
| ----- | ------ | ------------- |
| `hidden_layers` | Comma-separated integers | Sizes of hidden layers (e.g., `256,128` → two hidden layers) |
| `epochs` | Integer | Number of full passes over the training dataset |
| `learning_rate` | Float | SGD step size for weight updates |
| `batch_size` | Integer | Number of samples per gradient update (1 = online SGD) |

The input layer is always 784 (28×28 MNIST pixels, flattened) and the output layer is always 10 (digit classes 0–9).
