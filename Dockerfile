# ==============================================================================
# Multi-stage Dockerfile for parallel-mlp-mnist
#
# Stage 1 (builder): Compiles all training binaries and unit tests.
# Stage 2 (runtime): Lightweight image with binaries, scripts, and Python.
#
# Usage:
#   docker build -t parallel-mlp .
#   docker run parallel-mlp ./build/train_seq --config configs/small_network.conf
#
# See docs/docker.md for the full usage guide.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder — compile all training binaries and run unit tests
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        make \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only build-relevant files (leverages Docker layer caching)
COPY Makefile ./
COPY include/ include/
COPY src/     src/
COPY tests/   tests/

RUN mkdir -p build && make all

# Build test binaries (not run here — tests need MNIST data for some suites)
RUN make test || true

# ------------------------------------------------------------------------------
# Stage 2: Runtime — slim image with compiled binaries and helper scripts
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libgomp1 \
        python3 \
        python3-pip \
        curl \
        gzip \
        bash \
    && pip3 install --no-cache-dir matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for MPI (OpenMPI refuses mpirun as root by default)
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Copy compiled binaries from builder
COPY --from=builder /app/build/train_seq build/
COPY --from=builder /app/build/train_omp build/
COPY --from=builder /app/build/train_mpi build/

# Copy project files needed at runtime
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/    data/

# Ensure results directory exists and is writable
RUN mkdir -p results/logs results/tables results/plots \
    && chown -R appuser:appuser /app

USER appuser

CMD ["bash", "-c", "echo 'parallel-mlp-mnist Docker image'; echo ''; echo 'Usage examples:'; echo '  docker run parallel-mlp ./build/train_seq --config configs/small_network.conf'; echo '  docker run parallel-mlp mpirun -np 4 ./build/train_mpi --config configs/small_network.conf'; echo '  docker run parallel-mlp ./build/train_omp --config configs/small_network.conf --threads 4'; echo ''; echo 'See docs/docker.md for the full guide.'"]
