# Architecture

This document describes the code-level architecture of the project: module responsibilities, data structures, data flow through the network, and how the parallelization strategies are implemented in code. For the mathematical foundations (loss functions, activation equations, optimizer formulas), see [project_specification.md](project_specification.md).

---

## Module Overview

| Module | Header | Source | Responsibility | Key Types / Functions |
|--------|--------|--------|----------------|----------------------|
| **activations** | `activations.h` | `activations.cpp` | Activation functions | `relu()`, `relu_derivative()`, `softmax()` |
| **dataset** | `dataset.h` | `dataset.cpp` | MNIST IDX format loading | `Dataset`, `load_mnist_images()`, `load_mnist_labels()`, `free_dataset()` |
| **mlp** | `mlp.h` | `mlp.cpp` | Network structure and operations | `Layer`, `MLP`, `mlp_create()`, `mlp_forward()`, `mlp_backward()`, `mlp_update()`, `mlp_zero_gradients()`, `mlp_accumulate_gradients()`, `mlp_free()` |
| **loss** | `loss.h` | `loss.cpp` | Loss computation and gradients | `cross_entropy_loss()`, `cross_entropy_loss_batch()`, `cross_entropy_softmax_gradient()` |
| **optimizer** | `optimizer.h` | `optimizer.cpp` | SGD weight update | `sgd_update()` |
| **metrics** | `metrics.h` | `metrics.cpp` | Classification accuracy | `compute_accuracy()` |
| **timer** | `timer.h` | `timer.cpp` | Wall-clock timing | `Timer`, `timer_start()`, `timer_stop()`, `timer_elapsed_sec()` |
| **utils** | `utils.h` | `utils.cpp` | Config parsing and memory helpers | `Config`, `load_config()`, `alloc_vector()`, `alloc_matrix()`, `free_matrix()`, `he_init()`, `zero_init()` |

---

## Core Data Structures

### `Layer` (defined in `mlp.h`)

Each layer holds its own weights, biases, and all intermediate buffers needed for forward and backward passes.

| Field | Type | Dimensions | Description |
|-------|------|-----------|-------------|
| `weights` | `double*` | `[input_size × output_size]` | Weight matrix (row-major) |
| `biases` | `double*` | `[output_size]` | Bias vector |
| `z` | `double*` | `[output_size]` | Pre-activation values |
| `a` | `double*` | `[output_size]` | Post-activation values |
| `dW` | `double*` | `[input_size × output_size]` | Accumulated weight gradients |
| `db` | `double*` | `[output_size]` | Accumulated bias gradients |
| `delta` | `double*` | `[output_size]` | Error signal for backpropagation |
| `relu_d` | `double*` | `[output_size]` | Pre-allocated ReLU derivative buffer |
| `input_size` | `size_t` | — | Number of inputs to this layer |
| `output_size` | `size_t` | — | Number of neurons in this layer |

### `MLP` (defined in `mlp.h`)

| Field | Type | Description |
|-------|------|-------------|
| `layers` | `std::vector<Layer>` | All layers (hidden + output, excluding input) |
| `num_layers` | `size_t` | Number of layers |
| `input_size` | `size_t` | Network input dimension (784 for MNIST) |
| `output_size` | `size_t` | Network output dimension (10 for digit classes) |

### `Dataset` (defined in `dataset.h`)

| Field | Type | Description |
|-------|------|-------------|
| `images` | `double*` | Flattened pixel data `[num_samples × image_size]` |
| `labels` | `uint8_t*` | Class labels (0–9) `[num_samples]` |
| `num_samples` | `size_t` | Number of samples in this dataset |
| `image_size` | `size_t` | Pixels per image (784 for MNIST) |

### `Config` (defined in `utils.h`)

| Field | Type | Description |
|-------|------|-------------|
| `hidden_layers` | `std::vector<int>` | Hidden layer sizes (e.g., `{256, 128}`) |
| `epochs` | `int` | Number of training epochs |
| `learning_rate` | `double` | SGD step size |
| `batch_size` | `int` | Samples per gradient update (1 = online SGD) |

### `Timer` (defined in `timer.h`)

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | `chrono::high_resolution_clock::time_point` | Captured at `timer_start()` |
| `stop_time` | `chrono::high_resolution_clock::time_point` | Captured at `timer_stop()` |

---

## Data Flow

### Forward Pass — `mlp_forward(MLP& net, const double* input)`

Processes input through all layers sequentially. Returns a pointer to the output layer's activation buffer.

```
input (784) ──→ [Layer 0] ──→ [Layer 1] ──→ ... ──→ [Output Layer] ──→ softmax output (10)
                  W·x + b        W·a + b                W·a + b
                  ReLU           ReLU                    Softmax
```

For each layer:

1. **Linear transform**: `z[j] = biases[j] + Σ_k (prev_a[k] × weights[k × output_size + j])`
2. **Activation**: `relu(z, a, size)` for hidden layers; `softmax(z, a, size)` for the output layer

Weight indexing uses row-major order: element at row `k`, column `j` is accessed as `weights[k * output_size + j]`.

### Backward Pass — `mlp_backward(MLP& net, const double* input, uint8_t label)`

Computes gradients via backpropagation. Gradients **accumulate** into `dW`/`db` (not overwritten), enabling batch aggregation.

1. **Output layer delta**: `cross_entropy_softmax_gradient(a, label, delta, size)` — computes `δ_i = ŷ_i − 𝟙(i = label)`
2. **Hidden layer deltas** (from last to first): propagate error through transposed weights, gated by ReLU derivative:
   - `relu_derivative(z, relu_d, size)`
   - `δ_j = relu_d[j] × Σ_k (W_next[j × next_size + k] × δ_next[k])`
3. **Gradient accumulation** (all layers):
   - `dW[j × output_size + k] += prev_a[j] × delta[k]`
   - `db[k] += delta[k]`

### Training Step Lifecycle

```
mlp_zero_gradients(net)        ← Reset dW/db to zero
  ↓
for each sample in batch:
  mlp_forward(net, sample)     ← Compute predictions
  mlp_backward(net, sample)    ← Accumulate gradients
  ↓
scale dW/db by 1/batch_size    ← Average gradients
  ↓
mlp_update(net, learning_rate) ← Apply SGD: W -= lr × dW
```

---

## Parallelization Architecture

### OpenMP — `train_openmp.cpp`

Uses **shared-memory data parallelism** with thread-local network copies to avoid race conditions on gradient buffers.

```
Master Network (net)
  │
  ├──→ memcpy weights → thread_nets[0]  ──→  forward/backward on sample[s+0]
  ├──→ memcpy weights → thread_nets[1]  ──→  forward/backward on sample[s+1]
  ├──→ memcpy weights → thread_nets[2]  ──→  forward/backward on sample[s+2]
  └──→ memcpy weights → thread_nets[3]  ──→  forward/backward on sample[s+3]
                                                       │
                              mlp_accumulate_gradients() ← sum all thread dW/db
                                                       │
                                          scale by 1/batch_size
                                                       │
                                       mlp_update(net, learning_rate)
```

Key implementation details:

- **Thread-local copies**: `std::vector<MLP> thread_nets(num_threads)` — each thread gets its own `Layer` buffers for `z`, `a`, `dW`, `db`, `delta`, `relu_d`
- **Weight synchronization**: Before each batch, `std::memcpy` copies master weights/biases into all thread-local networks
- **Parallel region**: `#pragma omp parallel for reduction(+:batch_loss) schedule(static)` — distributes samples across threads; each thread accesses `thread_nets[omp_get_thread_num()]`
- **Gradient reduction**: After the parallel region, sequential loop calls `mlp_accumulate_gradients(net, thread_nets[t])` which performs element-wise `net.dW[j] += source.dW[j]` for all layers
- **Update**: Average gradients by `1.0 / actual_batch`, then `mlp_update()` on master network
- **Test evaluation**: Single-threaded on master network after each epoch

### MPI — `train_mpi.cpp`

Uses **distributed data parallelism** where each MPI rank owns a full network copy and processes a data shard, synchronizing gradients via collective operations.

```
Rank 0                    Rank 1                   Rank N-1
  │                         │                         │
  ├─ local samples ──→     ├─ local samples ──→      ├─ local samples ──→
  │  forward/backward      │  forward/backward       │  forward/backward
  │                         │                         │
  ├─ pack dW/db → local_grad                          │
  │                         │                         │
  └────────── MPI_Allreduce(MPI_SUM) ─────────────────┘
                            │
                   unpack → dW/db
                   scale by 1/batch_size
                   mlp_update()
                   (all ranks apply identical update)
```

Key implementation details:

- **Identical initialization**: All ranks call `mlp_create()` with the same seed (42), producing identical initial weights
- **Data partitioning**: Each rank computes `local_offset` and `local_count` from `rank`, `world_size`, and `actual_batch`; remainder samples go to lower-ranked processes
- **Gradient packing**: All `dW` and `db` arrays are serialized into a contiguous `local_grad` buffer (`double*` of `total_params` elements) via `std::memcpy`
- **Collective synchronization**: `MPI_Allreduce(local_grad, global_grad, total_params, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD)` — sums gradients across all ranks, result broadcast to all
- **Unpack and update**: `global_grad` is deserialized back into layer `dW`/`db`, scaled by `1.0 / actual_batch`, then `mlp_update()` is called — all ranks maintain identical weights
- **Loss aggregation**: `MPI_Reduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD)` — only rank 0 receives the total loss
- **Test evaluation**: Only rank 0 runs the forward pass on test data and prints statistics

---

## Memory Management

- **Allocation**: `alloc_vector(size)` and `alloc_matrix(rows, cols)` allocate raw `double*` arrays via `new double[]`
- **Deallocation**: `free_matrix()` calls `delete[]`; `mlp_free()` frees all per-layer buffers; `free_dataset()` frees image and label arrays
- **Matrix layout**: All weight matrices are stored as contiguous row-major flat arrays: `W[input_size × output_size]`, accessed as `W[row * output_size + col]`
- **Buffer lifecycle**: Gradient buffers (`dW`, `db`) are allocated once in `mlp_create()`, zeroed per batch via `mlp_zero_gradients()`, accumulated during `mlp_backward()`, and freed in `mlp_free()`