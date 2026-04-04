# MNIST Dataset

This directory holds the MNIST handwritten digit dataset in IDX binary format. The files must be downloaded before training.

## Expected Files

After downloading, `mnist/raw/` should contain:

| File | Contents | Size |
|------|----------|------|
| `train-images-idx3-ubyte` | 60,000 training images (28×28 pixels) | ~47 MB |
| `train-labels-idx1-ubyte` | 60,000 training labels (digits 0–9) | ~59 KB |
| `t10k-images-idx3-ubyte` | 10,000 test images (28×28 pixels) | ~7.8 MB |
| `t10k-labels-idx1-ubyte` | 10,000 test labels (digits 0–9) | ~10 KB |

## IDX Binary Format

MNIST files use the IDX binary format (big-endian byte order). Each file begins with a magic number header that encodes the data type and number of dimensions, followed by dimension sizes (4 bytes each), then the raw data bytes.

- **Image files**: magic `0x00000803` → 3 dimensions (count, rows, cols) → unsigned byte pixel values (0–255)
- **Label files**: magic `0x00000801` → 1 dimension (count) → unsigned byte labels (0–9)

Pixel values are normalized to `[0.0, 1.0]` at load time by the dataset loader.

> These files are `.gitignore`d and will not be committed to the repository. See the root [README.md](../README.md) for download instructions.