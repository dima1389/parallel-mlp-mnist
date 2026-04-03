#!/usr/bin/env bash
# Download and extract MNIST dataset files into data/mnist/raw/.
# Downloads four gzipped IDX files from the official source and decompresses them.
# Skips files that already exist to allow re-running safely.
set -euo pipefail

BASE_URL="https://yann.lecun.com/exdb/mnist"
OUT_DIR="data/mnist/raw"

mkdir -p "$OUT_DIR"

# The four MNIST data files (images + labels for train and test splits)
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    if [ ! -f "$OUT_DIR/${f%.gz}" ]; then
        echo "Downloading $f ..."
        curl -L -o "$OUT_DIR/$f" "$BASE_URL/$f"
        echo "Extracting $f ..."
        gunzip -f "$OUT_DIR/$f"
    else
        echo "$OUT_DIR/${f%.gz} already exists, skipping."
    fi
done

echo "MNIST download complete."
