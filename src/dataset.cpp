#include "dataset.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

// Read a 32-bit big-endian integer from a binary stream.
// MNIST IDX files store all header values in big-endian format.
static uint32_t read_u32_be(std::ifstream& f) {
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8)  |
           (static_cast<uint32_t>(buf[3]));
}

// Load MNIST image file (IDX3 format).
// File structure: magic(2051) | num_images | rows | cols | pixel_bytes...
// Returns normalized pixel data as doubles in [0, 1].
Dataset load_mnist_images(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open image file: " + filepath);
    }

    // Validate magic number for IDX3 (image) format
    uint32_t magic = read_u32_be(f);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    // Read image dimensions from header
    uint32_t num_images = read_u32_be(f);
    uint32_t rows       = read_u32_be(f);
    uint32_t cols       = read_u32_be(f);

    size_t image_size = static_cast<size_t>(rows) * cols;
    size_t total_pixels = static_cast<size_t>(num_images) * image_size;

    // Read raw pixel bytes (uint8)
    auto* raw = new uint8_t[total_pixels];
    f.read(reinterpret_cast<char*>(raw), static_cast<std::streamsize>(total_pixels));

    // Convert to double and normalize pixel values from [0,255] to [0,1]
    auto* images = new double[total_pixels];
    for (size_t i = 0; i < total_pixels; ++i) {
        images[i] = static_cast<double>(raw[i]) / 255.0;
    }
    delete[] raw;

    Dataset ds;
    ds.images      = images;
    ds.labels      = nullptr;
    ds.num_samples = num_images;
    ds.image_size  = image_size;
    return ds;
}

// Load MNIST label file (IDX1 format).
// File structure: magic(2049) | num_labels | label_bytes...
Dataset load_mnist_labels(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open label file: " + filepath);
    }

    // Validate magic number for IDX1 (label) format
    uint32_t magic = read_u32_be(f);
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_u32_be(f);

    // Each label is a single byte (0-9)
    auto* labels = new uint8_t[num_labels];
    f.read(reinterpret_cast<char*>(labels), static_cast<std::streamsize>(num_labels));

    Dataset ds;
    ds.images      = nullptr;
    ds.labels      = labels;
    ds.num_samples = num_labels;
    ds.image_size  = 0;
    return ds;
}

// Release heap-allocated arrays and reset all fields.
void free_dataset(Dataset& ds) {
    delete[] ds.images;
    delete[] ds.labels;
    ds.images      = nullptr;
    ds.labels      = nullptr;
    ds.num_samples = 0;
    ds.image_size  = 0;
}
