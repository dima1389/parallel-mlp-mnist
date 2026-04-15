# Network Configurations

This directory contains configuration files that define MLP network architecture and training hyperparameters.

## Configuration File Format

Configuration files use a simple `key=value` format:

```ini
hidden_layers=256,128
epochs=10
learning_rate=0.01
batch_size=64
```

| Key | Type | Description |
|-----|------|-------------|
| `hidden_layers` | Comma-separated integers | Sizes of hidden layers (e.g., `256,128` → two hidden layers) |
| `epochs` | Integer | Number of full passes over the training dataset |
| `learning_rate` | Float | SGD step size for weight updates |
| `batch_size` | Integer | Number of samples per gradient update |

The input layer is always 784 (28×28 MNIST pixels, flattened) and the output layer is always 10 (digit classes 0–9). These are not configurable.

## Predefined Configurations

| Config File | Hidden Layers | Architecture | ~Parameters | Epochs |
|-------------|---------------|--------------|-------------|--------|
| `small_network.conf` | 128 | 784 → 128 → 10 | 101,770 | 10 |
| `medium_network.conf` | 256, 128 | 784 → 256 → 128 → 10 | 234,890 | 10 |
| `large_network.conf` | 512, 256, 128 | 784 → 512 → 256 → 128 → 10 | 467,338 | 10 |
| `test_benchmark.conf` | 128 | 784 → 128 → 10 | 101,770 | 1 |

All predefined configurations use: `learning_rate=0.01`, `batch_size=64`.

> **MPI note:** When training with MPI, each mini-batch is split evenly across ranks.
> If `batch_size` is smaller than the number of MPI processes, some ranks will receive
> zero samples per batch. Use `batch_size >= world_size` for proper load distribution.

The `test_benchmark.conf` is a single-epoch variant of `small_network.conf`, used for quick validation of the benchmark infrastructure.
