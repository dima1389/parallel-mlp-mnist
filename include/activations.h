#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// Activation functions used in the MLP hidden and output layers.
// ReLU is applied to hidden layers; Softmax to the output layer.

#include <cstddef>

// ReLU activation: out[i] = max(0, in[i]). Zeroes out negative values.
void relu(const double* input, double* output, size_t size);

// ReLU derivative: returns 1.0 for positive inputs, 0.0 otherwise.
// Used during backpropagation to compute hidden layer gradients.
void relu_derivative(const double* input, double* output, size_t size);

// Softmax activation: converts raw logits into a probability distribution.
// out[i] = exp(in[i]) / sum(exp(in[j])), applied to a single sample vector.
void softmax(const double* input, double* output, size_t size);

#endif // ACTIVATIONS_H
