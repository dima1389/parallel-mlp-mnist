#include "dataset.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

// Integration tests for MNIST dataset loading.
// Requires actual MNIST files in data/mnist/raw/ (download via scripts/download_mnist.sh).

// Verify that image loading reads the correct number of samples
// and that pixel values are normalized to [0, 1].
static void test_load_images() {
    const char* path = "data/mnist/raw/train-images-idx3-ubyte";
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("  SKIP: %s not found (run download_mnist first)\n", path);
        return;
    }
    std::fclose(f);

    Dataset ds = load_mnist_images(path);
    assert(ds.num_samples == 60000);    // MNIST training set has 60k images
    assert(ds.image_size == 784);       // 28 * 28 pixels

    // Spot-check: first 100 pixel values should be in [0, 1]
    for (size_t i = 0; i < 100; ++i) {
        assert(ds.images[i] >= 0.0 && ds.images[i] <= 1.0);
    }

    free_dataset(ds);
    std::printf("  PASS: load_mnist_images (%zu samples, %zu pixels)\n",
                ds.num_samples, ds.image_size);
}

// Verify that label loading reads the correct number of labels
// and that all values are valid digit classes (0-9).
static void test_load_labels() {
    const char* path = "data/mnist/raw/train-labels-idx1-ubyte";
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("  SKIP: %s not found (run download_mnist first)\n", path);
        return;
    }
    std::fclose(f);

    Dataset ds = load_mnist_labels(path);
    assert(ds.num_samples == 60000);  // 60k labels matching 60k images

    // Every label must be a valid digit (0-9)
    for (size_t i = 0; i < ds.num_samples; ++i) {
        assert(ds.labels[i] <= 9);
    }

    free_dataset(ds);
    std::printf("  PASS: load_mnist_labels (%zu labels)\n", ds.num_samples);
}

int main() {
    std::printf("=== test_dataset ===\n");
    test_load_images();
    test_load_labels();
    std::printf("All dataset tests passed (or skipped if data unavailable).\n");
    return 0;
}
