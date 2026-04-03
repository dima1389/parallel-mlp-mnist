# Project Specification: Parallel MLP Training on MNIST

## 1. Overview

This project implements a **Multilayer Perceptron (MLP)** neural network for handwritten digit classification on the MNIST dataset. Three training implementations are provided—**sequential**, **OpenMP**, and **MPI**—enabling direct comparison of training performance across parallelization strategies and varying network sizes.

The primary objective is **training time optimization** through parallelism; classification accuracy is a secondary concern.

## 2. Problem Statement

Train and evaluate an MLP classifier on the MNIST dataset using standard gradient descent. Compare wall-clock training time between sequential and parallel implementations for different network architectures and parallelism levels (number of threads/processes), and report speedup metrics.

## 3. Inputs

The program accepts the following inputs via command-line arguments and configuration files:

| Input | Source | Description |
|-------|--------|-------------|
| Network architecture | Config file (`hidden_layers`) | Comma-separated list of hidden layer sizes (e.g., `256,128`) |
| Number of epochs | Config file (`epochs`) | Training iterations over the full dataset |
| Learning rate | Config file (`learning_rate`) | Step size for SGD weight updates |
| Batch size | Config file (`batch_size`) | Mini-batch size for gradient averaging |
| Data directory | CLI argument | Path to MNIST raw binary files |
| Thread/process count | Environment / CLI | `OMP_NUM_THREADS` for OpenMP; `-np` for MPI via `mpirun` |

### 3.1 Configuration Files

Three predefined network configurations are provided in configs:

| Config | Architecture | Parameters |
|--------|-------------|------------|
| small_network.conf | 784 → 128 → 10 | ~100K weights |
| medium_network.conf | 784 → 256 → 128 → 10 | ~234K weights |
| large_network.conf | 784 → 512 → 256 → 128 → 10 | ~467K weights |

All configurations use 10 epochs, learning rate 0.01, and batch size 1 (online SGD).

### 3.2 Dataset

**MNIST** handwritten digits in IDX binary format:

- **Training set**: 60,000 grayscale images (28×28 pixels) with labels (0–9)
- **Test set**: 10,000 grayscale images with labels
- Pixel values normalized from [0, 255] to [0.0, 1.0] at load time
- Download scripts provided for both Linux (download_mnist.sh) and Windows (download_mnist.ps1)

## 4. Neural Network Architecture

### 4.1 Structure

The MLP consists of an input layer (784 neurons), one or more configurable hidden layers, and a fixed output layer (10 neurons for digit classes 0–9).

### 4.2 Activation Functions

- **Hidden layers**: ReLU — `f(x) = max(0, x)`
- **Output layer**: Softmax with numerical stability (subtracts max before exponentiation)

### 4.3 Weight Initialization

- **Weights**: He initialization — sampled from `N(0, √(2/fan_in))` with fixed seed 42 for reproducibility across all implementations
- **Biases**: Zero-initialized

### 4.4 Loss Function

**Cross-entropy loss** with clamped probabilities (minimum 1e-12) to prevent numerical issues:

$$L = -\log(\hat{y}_{label})$$

The combined softmax–cross-entropy gradient is used for efficient backpropagation:

$$\nabla_i = \hat{y}_i - \mathbb{1}(i = label)$$

### 4.5 Optimizer

**Vanilla Stochastic Gradient Descent (SGD)**:

$$w \leftarrow w - \eta \cdot \nabla w$$

No momentum, weight decay, or adaptive learning rate is used.

### 4.6 Data Layout

All matrices use **row-major flat arrays** (`double*`) for cache-friendly access. Weight matrices are stored as `[input_size × output_size]`. Each layer maintains separate buffers for pre-activations (`z`), activations (`a`), weight gradients (`dW`), bias gradients (`db`), and error signals (`delta`).

## 5. Training Implementations

### 5.1 Sequential (`train_sequential.cpp`)

Baseline single-threaded implementation using **online SGD** (one sample at a time):

1. For each epoch, iterate over all 60,000 training samples
2. Per sample: zero gradients → forward pass → compute loss → backward pass → update weights
3. After each epoch: evaluate classification accuracy on the test set
4. Report per-epoch training loss, test accuracy, and wall-clock time

### 5.2 OpenMP (`train_openmp.cpp`)

**Shared-memory data parallelism** using thread-local network copies:

1. Create `N` thread-local copies of the master network (where `N` = number of OpenMP threads)
2. Process training data in mini-batches of size `N` (one sample per thread)
3. Per mini-batch step:
   - **Sync**: Copy master weights to all thread-local networks; zero local gradients
   - **Parallel region** (`#pragma omp parallel for`): Each thread computes forward pass, loss, and backward pass on its assigned sample using its local network copy
   - **Aggregate**: Sum all thread-local gradients into the master network
   - **Average**: Divide accumulated gradients by the actual batch size
   - **Update**: Apply SGD to master network weights
4. After each epoch: evaluate test accuracy (single-threaded)

### 5.3 MPI (`train_mpi.cpp`)

**Distributed data parallelism** using gradient synchronization via `MPI_Allreduce`:

1. All ranks load the full dataset independently (avoids MPI scatter complexity)
2. Partition training samples across ranks: each rank computes `local_start` and `local_count` based on its rank and world size (remainder samples distributed round-robin to first ranks)
3. Per sample in each rank's local partition:
   - Compute forward pass and backward pass locally
   - **Pack** all weight and bias gradients into a flat buffer
   - **`MPI_Allreduce`** with `MPI_SUM` to sum gradients across all ranks
   - **Unpack** and divide by world size
   - Apply SGD update (all ranks remain synchronized)
4. After each epoch: only rank 0 evaluates test accuracy and prints statistics

### 5.4 Comparison of Parallelization Strategies

| Aspect | Sequential | OpenMP | MPI |
|--------|-----------|--------|-----|
| Parallelism | None | Shared memory, multi-threaded | Distributed, multi-process |
| Data division | Full dataset, sequential | Mini-batch per sync step | Full shard per rank |
| Gradient computation | Per-sample | Per-thread (parallel) | Per-rank (local) |
| Gradient aggregation | Direct update | Sum thread-local → master | `MPI_Allreduce` (SUM) |
| Synchronization | N/A | Implicit barrier at batch boundary | Explicit Allreduce per sample |
| Network copies | 1 | 1 master + N thread-local | 1 per rank (kept identical) |

## 6. Build System

The project uses GNU Make with the following targets:

| Target | Command | Output | Notes |
|--------|---------|--------|-------|
| Sequential | `make seq` | `build/train_seq` | Default target, uses `g++` |
| OpenMP | `make omp` | `build/train_omp` | Adds `-fopenmp` flag |
| MPI | `make mpi` | `build/train_mpi` | Uses `mpicxx` compiler wrapper |
| Tests | `make test` | `build/test_*` | Builds and runs all tests |
| Clean | `make clean` | — | Removes build directory |

**Compiler flags**: `-std=c++17 -O2 -Wall -Wextra -Iinclude`

Object files use suffixes (`_omp`, `_mpi`) to avoid conflicts between build variants.

## 7. Testing

Unit tests in tests validate core components:

- **test_dataset.cpp** — MNIST loading: correct sample count, image dimensions, pixel normalization [0,1], label range [0–9]
- **test_loss.cpp** — ReLU forward/derivative, softmax normalization and numerical stability with large inputs, cross-entropy loss correctness, gradient formula verification
- **test_mlp.cpp** — Network creation/destruction, forward pass output shape and softmax properties, gradient non-zero after backward pass, loss reduction over 50 SGD steps

## 8. Benchmarking and Results

### 8.1 Benchmark Script

benchmark.sh automates the full benchmark suite:

- Runs all three configurations (small, medium, large)
- For each: sequential baseline, then OpenMP and MPI with 1, 2, 4, and 8 threads/processes
- Extracts last-epoch wall-clock time from program output
- Writes results to a timestamped CSV file in tables

### 8.2 Results Processing

plot_results.py processes benchmark CSVs:

- Generates ASCII tables with per-configuration timing and speedup values
- Creates bar chart PNGs (`results/plots/speedup_{config}.png`) using matplotlib
- Speedup calculated as: `sequential_time / parallel_time`

### 8.3 Expected Results Format

| Implementation | Training Duration [s] | Speedup |
|:-:|:-:|:-:|
| Sequential | — | 1.00 |
| OpenMP (2 threads) | — | — |
| OpenMP (4 threads) | — | — |
| OpenMP (8 threads) | — | — |
| MPI (2 processes) | — | — |
| MPI (4 processes) | — | — |
| MPI (8 processes) | — | — |

One table per network configuration (small, medium, large).

## 9. Project Structure

```
parallel-mlp-mnist/
├── Makefile                     # Build system (seq, omp, mpi, test, clean)
├── README.md                    # Project overview
├── configs/                     # Network configuration files
│   ├── small_network.conf       # 784→128→10
│   ├── medium_network.conf      # 784→256→128→10
│   └── large_network.conf       # 784→512→256→128→10
├── data/mnist/raw/              # MNIST binary files (IDX format)
├── docs/
│   ├── project_specification.md # This document
│   └── architecture.md          # Architecture documentation
├── include/                     # Header files
│   ├── activations.h            # ReLU, softmax
│   ├── dataset.h                # MNIST loading (Dataset struct)
│   ├── loss.h                   # Cross-entropy loss and gradient
│   ├── metrics.h                # Classification accuracy
│   ├── mlp.h                    # MLP network (Layer/MLP structs, forward/backward)
│   ├── optimizer.h              # SGD update rule
│   ├── timer.h                  # High-resolution timing
│   └── utils.h                  # Config parsing, memory allocation, initialization
├── results/
│   ├── logs/                    # Training logs
│   ├── plots/                   # Speedup bar charts
│   └── tables/                  # Benchmark CSV files
├── scripts/
│   ├── benchmark.sh             # Automated benchmark runner
│   ├── download_mnist.sh        # Dataset download (Linux)
│   ├── download_mnist.ps1       # Dataset download (Windows)
│   └── plot_results.py          # Results visualization
├── src/                         # Implementation source files
│   ├── train_sequential.cpp     # Sequential training loop
│   ├── train_openmp.cpp         # OpenMP parallel training
│   ├── train_mpi.cpp            # MPI distributed training
│   ├── mlp.cpp                  # Core network operations
│   ├── activations.cpp          # Activation functions
│   ├── dataset.cpp              # MNIST data loading
│   ├── loss.cpp                 # Loss computation
│   ├── metrics.cpp              # Accuracy evaluation
│   ├── optimizer.cpp            # SGD implementation
│   ├── timer.cpp                # Timing utilities
│   └── utils.cpp                # Config parsing, allocation helpers
└── tests/                       # Unit tests
    ├── test_dataset.cpp
    ├── test_loss.cpp
    └── test_mlp.cpp
```

## 10. Key Design Decisions

1. **Reproducibility**: Fixed random seed (42) for He initialization ensures identical initial weights across all implementations
2. **Numerical safety**: Softmax uses max-subtraction; cross-entropy clamps probabilities at 1e-12
3. **Row-major flat arrays**: All matrices stored as contiguous `double*` for cache efficiency and straightforward MPI serialization
4. **Full dataset on all ranks (MPI)**: Each rank loads the complete dataset to avoid scatter/gather complexity with IDX binary format
5. **Online SGD**: All configs use batch_size=1, emphasizing per-sample parallelism and frequent weight updates
6. **Gradient accumulation pattern**: Backward pass accumulates into persistent `dW`/`db` buffers, enabling flexible batch aggregation across threads or processes
