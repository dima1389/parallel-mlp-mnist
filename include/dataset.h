#ifndef DATASET_H
#define DATASET_H

// MNIST dataset loading and memory management.
// Reads IDX binary files and normalizes pixel values to [0, 1].

#include <cstddef>
#include <cstdint>
#include <string>
#include "types.h"

// Holds either image data or label data (or both) for an MNIST split.
struct Dataset {
    real_t*  images;       // Flattened pixel data: [num_samples x image_size]
    uint8_t* labels;       // Class labels (0-9): [num_samples]
    size_t   num_samples;  // Number of samples in this dataset
    size_t   image_size;   // Pixels per image (28*28 = 784 for MNIST)
};

// Load MNIST image file (IDX3 format). Returns Dataset with images populated,
// labels set to nullptr. Pixel values are normalized from [0,255] to [0,1].
Dataset load_mnist_images(const std::string& filepath);

// Load MNIST label file (IDX1 format). Returns Dataset with labels populated,
// images set to nullptr.
Dataset load_mnist_labels(const std::string& filepath);

// Free all heap-allocated memory in a Dataset and reset fields to zero/nullptr.
void free_dataset(Dataset& ds);

#endif // DATASET_H
