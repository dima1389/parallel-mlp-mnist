# Docker Setup Guide

This document explains how to build, run, and develop the parallel-mlp-mnist project using Docker. Docker eliminates the need to manually install g++, OpenMP, MPI, Python, and other dependencies — everything is bundled into a single container image.

---

## Table of Contents

- [Docker Setup Guide](#docker-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Image Architecture](#image-architecture)
  - [Docker Compose Services](#docker-compose-services)
    - [download-data](#download-data)
    - [train-seq](#train-seq)
    - [train-omp](#train-omp)
    - [train-mpi](#train-mpi)
    - [benchmark](#benchmark)
    - [test-benchmark](#test-benchmark)
    - [test](#test)
    - [plot](#plot)
  - [Environment Variables](#environment-variables)
  - [Volume Mounts](#volume-mounts)
  - [Using Docker Directly (without Compose)](#using-docker-directly-without-compose)
  - [VS Code Dev Container](#vs-code-dev-container)
  - [CI / GitHub Actions](#ci--github-actions)
  - [Troubleshooting](#troubleshooting)
  - [Future Enhancements](#future-enhancements)

---

## Prerequisites

| Tool                                                         | Version | Purpose                                                      |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| [Docker](https://docs.docker.com/get-docker/) | 20.10+ | Container runtime |
| [Docker Compose](https://docs.docker.com/compose/install/) | v2+ | Multi-service orchestration (bundled with Docker Desktop) |

No other tools are needed. The Docker image includes g++, OpenMPI, Python 3, matplotlib, and all build dependencies.

---

## Quick Start

```bash
# 1. Build the Docker image
docker build -t parallel-mlp .

# 2. Download the MNIST dataset (one-time)
docker compose run download-data

# 3. Train with the sequential baseline
docker compose run train-seq

# 4. Train with OpenMP (4 threads)
THREADS=4 docker compose run train-omp

# 5. Train with MPI (4 processes)
PROCS=4 docker compose run train-mpi

# 6. Run the full benchmark suite
docker compose run benchmark

# 7. Generate plots from results
docker compose run plot
```

Results appear in the `results/` directory on your host machine.

---

## Image Architecture

The project uses a **multi-stage Docker build** defined in `Dockerfile`:

```text
┌─────────────────────────────────┐
│  Stage 1: builder (ubuntu:22.04)│
│  ┌────────────────────────────┐ │
│  │ g++, make, libopenmpi-dev  │ │
│  │ Compiles: train_seq,       │ │
│  │   train_omp, train_mpi    │ │
│  └────────────────────────────┘ │
└──────────────┬──────────────────┘
               │ COPY binaries
               ▼
┌─────────────────────────────────┐
│  Stage 2: runtime (ubuntu:22.04)│
│  ┌────────────────────────────┐ │
│  │ openmpi-bin, libgomp1,     │ │
│  │ python3, matplotlib        │ │
│  │ + compiled binaries        │ │
│  │ + scripts, configs         │ │
│  └────────────────────────────┘ │
│  USER: appuser (non-root)       │
└─────────────────────────────────┘
```

**Why multi-stage?** The builder stage includes `g++`, `make`, and `libopenmpi-dev` (~300 MB of headers/libraries) needed only for compilation. The runtime stage contains only what's needed to execute: the compiled binaries, OpenMPI runtime, and Python for plotting. This cuts the final image size significantly.

**Why non-root?** OpenMPI refuses to run as root by default (security policy). The runtime image creates a dedicated `appuser` to run MPI safely without `--allow-run-as-root`.

---

## Docker Compose Services

### download-data

Downloads the four MNIST binary files into a persistent named volume.

```bash
docker compose run download-data
```

This only needs to run once. The `mnist-data` volume persists across container restarts.

### train-seq

Trains the MLP using the sequential (single-threaded) implementation.

```bash
docker compose run train-seq                                            # default: small_network.conf
CONFIG=configs/large_network.conf docker compose run train-seq          # custom config
```

### train-omp

Trains the MLP using OpenMP shared-memory parallelism.

```bash
THREADS=4 docker compose run train-omp                                  # 4 threads
THREADS=8 CONFIG=configs/medium_network.conf docker compose run train-omp
```

### train-mpi

Trains the MLP using MPI distributed parallelism.

```bash
PROCS=4 docker compose run train-mpi                                    # 4 processes
PROCS=8 CONFIG=configs/large_network.conf docker compose run train-mpi
```

> **Note**: The compose service uses `--allow-run-as-root` because the container user context may vary. When using docker directly with `USER appuser`, the flag is not needed.

### benchmark

Runs the complete benchmark suite across all configurations and parallelism levels.

```bash
docker compose run benchmark                                            # single run
NUM_RUNS=5 docker compose run benchmark                                 # 5 repeated runs
```

### test-benchmark

Runs a quick 1-epoch validation to verify the pipeline works.

```bash
docker compose run test-benchmark
```

### test

Builds and runs all unit tests. Uses the `builder` stage (which has the compiler) so it can recompile test binaries.

```bash
docker compose run test
```

### plot

Re-generates summary tables and PNG charts from existing benchmark results.

```bash
docker compose run plot
```

---

## Environment Variables

| Variable   | Default                       | Used By                      | Description                   |
| ---------- | ----------------------------- | ---------------------------- | ----------------------------- |
| `CONFIG` | `configs/small_network.conf` | train-seq, train-omp, train-mpi | Network configuration file |
| `THREADS` | `4` | train-omp | Number of OpenMP threads |
| `PROCS` | `4` | train-mpi | Number of MPI processes |
| `NUM_RUNS` | `1` | benchmark, test-benchmark | Number of repeated training runs |

Set variables inline before the `docker compose` command:

```bash
THREADS=8 CONFIG=configs/large_network.conf NUM_RUNS=3 docker compose run train-omp
```

---

## Volume Mounts

| Mount                          | Container Path       | Purpose                                                      |
| ------------------------------ | -------------------- | ------------------------------------------------------------ |
| `mnist-data` (named volume) | `/app/data/mnist/raw` | Persistent MNIST dataset (survives container removal) |
| `./results` (bind mount) | `/app/results` | Training logs, CSVs, and plots (visible on host) |
| `./configs` (bind mount) | `/app/configs` | Config files (edit on host, changes are immediate) |

---

## Using Docker Directly (without Compose)

If you prefer not to use Docker Compose:

```bash
# Build the image
docker build -t parallel-mlp .

# Download MNIST into a named volume
docker run -v mnist-data:/app/data/mnist/raw parallel-mlp bash scripts/download_mnist.sh

# Sequential training
docker run \
  -v mnist-data:/app/data/mnist/raw \
  -v ./results:/app/results \
  parallel-mlp \
  ./build/train_seq --config configs/small_network.conf

# OpenMP training (4 threads)
docker run \
  -e OMP_NUM_THREADS=4 \
  -v mnist-data:/app/data/mnist/raw \
  -v ./results:/app/results \
  parallel-mlp \
  ./build/train_omp --config configs/small_network.conf --threads 4

# MPI training (4 processes) — needs --allow-run-as-root or use appuser
docker run \
  -v mnist-data:/app/data/mnist/raw \
  -v ./results:/app/results \
  parallel-mlp \
  mpirun --allow-run-as-root -np 4 ./build/train_mpi --config configs/small_network.conf

# Run unit tests (using builder stage)
docker build --target builder -t parallel-mlp-builder .
docker run -v mnist-data:/app/data/mnist/raw parallel-mlp-builder make test

# Generate plots
docker run \
  -v ./results:/app/results \
  parallel-mlp \
  python3 scripts/plot_results.py
```

---

## VS Code Dev Container

The project includes a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) configuration for a full development environment inside Docker.

### Setup

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code
2. Open the project folder in VS Code
3. Press `F1` → **Dev Containers: Reopen in Container**
4. VS Code rebuilds the container and opens a terminal inside it

### What's Included

The dev container (`.devcontainer/Dockerfile`) is a **single-stage** image with:

- **Build tools**: g++, make, gdb, valgrind
- **MPI**: libopenmpi-dev, openmpi-bin (with root-as-root enabled for convenience)
- **Python**: python3, matplotlib
- **Utilities**: git, curl, vim

On first open, `make all` runs automatically (`postCreateCommand`) to compile all binaries.

### Developing

Once inside the container, the full toolchain is available:

```bash
make all                          # Rebuild binaries
make test                         # Run unit tests
bash scripts/download_mnist.sh    # Download MNIST (if not already present)
./build/train_seq --config configs/small_network.conf
./build/train_omp --config configs/small_network.conf --threads 4
mpiexec -np 4 ./build/train_mpi --config configs/small_network.conf
bash scripts/benchmark.sh         # Full benchmark
```

---

## CI / GitHub Actions

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push to `main` and on pull requests:

1. **Build** the Docker image
2. **Download** the MNIST dataset
3. **Run unit tests** (`make test`)
4. **Run a quick validation benchmark** (`scripts/test_benchmark.sh`)
5. **Upload** benchmark results as a CI artifact

Benchmark CSVs and plots are available for download from the workflow run's Artifacts section.

---

## Troubleshooting

### MPI: "Running as root is not allowed"

OpenMPI refuses to run as root for security reasons. Solutions:

- **Docker Compose**: The `train-mpi` service already includes `--allow-run-as-root`
- **Docker directly**: Add `--allow-run-as-root` to the `mpirun` command
- **Dev Container**: The `OMPI_ALLOW_RUN_AS_ROOT=1` env var is set in the dev Dockerfile

### Volume permission errors

If results can't be written to `./results/`:

```bash
# Fix host directory permissions
chmod -R 777 results/

# Or run the container as your host user
docker run --user "$(id -u):$(id -g)" ...
```

### MNIST download fails inside container

If `curl` can't reach the MNIST server, download on the host first and bind-mount:

```bash
bash scripts/download_mnist.sh    # download on host
docker run -v ./data/mnist/raw:/app/data/mnist/raw ...
```

### Slow Docker builds

The Dockerfile leverages layer caching. If only source files change, the `apt-get` layer is cached. To force a full rebuild:

```bash
docker build --no-cache -t parallel-mlp .
```

---

## Future Enhancements

- **Multi-arch images**: Build for `linux/amd64` and `linux/arm64` using `docker buildx` for Apple Silicon / ARM servers
- **GHCR publishing**: Push images to GitHub Container Registry so users can `docker pull` instead of building locally
- **CUDA support**: Add a GPU-accelerated build stage with NVIDIA CUDA base image
- **Multi-node MPI**: Docker Compose with multiple containers communicating over a Docker network (requires SSH + hostfiles)
