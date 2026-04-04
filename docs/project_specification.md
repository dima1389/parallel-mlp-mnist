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

> For code-level implementation details (data structures, buffer management, parallelization mechanics), see [architecture.md](architecture.md).

### 5.1 Sequential (`train_sequential.cpp`)

Baseline single-threaded implementation using **online SGD** (one sample at a time):

1. For each epoch, iterate over all 60,000 training samples
2. Per sample: zero gradients → forward pass → compute loss → backward pass → update weights
3. After each epoch: evaluate classification accuracy on the test set
4. Report per-epoch training loss, test accuracy, and wall-clock time

### 5.2 OpenMP (`train_openmp.cpp`)

**Shared-memory data parallelism** across training samples:

1. Create `N` independent network copies (where `N` = number of threads)
2. Process training data in mini-batches of size `N` (one sample per thread)
3. Per mini-batch step:
   - **Sync**: Propagate master weights to all copies; zero local gradients
   - **Parallel compute**: Each thread computes forward pass, loss, and backward pass on its assigned sample independently
   - **Aggregate**: Sum all local gradients into the master network
   - **Average**: Divide accumulated gradients by the actual batch size
   - **Update**: Apply SGD to master network weights
4. After each epoch: evaluate test accuracy (single-threaded)

### 5.3 MPI (`train_mpi.cpp`)

**Distributed data parallelism** using collective gradient synchronization:

1. All ranks load the full dataset independently (avoids scatter complexity)
2. Partition training samples across ranks (remainder samples distributed round-robin to lower-ranked processes)
3. Per sample in each rank's local partition:
   - Compute forward pass and backward pass locally
   - Collectively sum gradients across all ranks
   - Average gradients by the batch size
   - Apply SGD update (all ranks apply the identical update, keeping weights synchronized)
4. After each epoch: only rank 0 evaluates test accuracy and prints statistics

### 5.4 Comparison of Parallelization Strategies

| Aspect | Sequential | OpenMP | MPI |
|--------|-----------|--------|-----|
| Parallelism | None | Shared memory, multi-threaded | Distributed, multi-process |
| Data division | Full dataset, sequential | Mini-batch per sync step | Full shard per rank |
| Gradient computation | Per-sample | Per-thread (parallel) | Per-rank (local) |
| Gradient aggregation | Direct update | Sum local → master | Collective sum across ranks |
| Synchronization | N/A | Implicit barrier at batch boundary | Explicit collective operation per sample |
| Network copies | 1 | 1 master + N local | 1 per rank (kept identical) |

## 6. Expected Results Format

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

## 7. Key Design Decisions

1. **Reproducibility**: Fixed random seed (42) for He initialization ensures identical initial weights across all implementations
2. **Numerical safety**: Softmax uses max-subtraction; cross-entropy clamps probabilities at 1e-12
3. **Row-major flat arrays**: All matrices stored as contiguous `double*` for cache efficiency and straightforward MPI serialization
4. **Full dataset on all ranks (MPI)**: Each rank loads the complete dataset to avoid scatter/gather complexity with IDX binary format
5. **Online SGD**: All configs use batch_size=1, emphasizing per-sample parallelism and frequent weight updates
6. **Gradient accumulation pattern**: Backward pass accumulates into persistent `dW`/`db` buffers, enabling flexible batch aggregation across threads or processes
