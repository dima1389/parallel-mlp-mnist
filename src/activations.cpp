#include "activations.h"
#include <cmath>
#include <algorithm>

// Apply ReLU element-wise: pass through positive values, zero out negatives.
void relu(const real_t* input, real_t* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0) ? input[i] : 0.0;
    }
}

// Compute ReLU derivative element-wise: 1 if input > 0, else 0.
void relu_derivative(const real_t* input, real_t* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = (input[i] > 0.0) ? 1.0 : 0.0;
    }
}

// Compute softmax with numerical stability (subtract max to prevent overflow).
void softmax(const real_t* input, real_t* output, size_t size) {
    // Find max value to shift inputs and avoid exp() overflow
    real_t max_val = input[0];
    for (size_t i = 1; i < size; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute shifted exponentials and their sum
    real_t sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probability distribution
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}
