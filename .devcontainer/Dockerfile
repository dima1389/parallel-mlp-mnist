# Development container for parallel-mlp-mnist
#
# Single-stage image with both build tools and runtime dependencies.
# Used by VS Code Dev Containers for a full development environment.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build tools
        g++ \
        make \
        gdb \
        valgrind \
        # MPI (compiler + runtime)
        libopenmpi-dev \
        openmpi-bin \
        # Python + plotting
        python3 \
        python3-pip \
        # Data download
        curl \
        gzip \
        # Utilities
        bash \
        git \
        less \
        vim \
    && pip3 install --no-cache-dir matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Allow OpenMPI to run as root inside devcontainer (common for dev)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /workspace
