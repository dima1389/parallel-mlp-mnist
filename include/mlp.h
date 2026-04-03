#ifndef MLP_H
#define MLP_H

// Multi-Layer Perceptron (MLP) neural network implementation.
// Supports configurable hidden layers with ReLU activation and
// softmax output for multi-class classification (e.g., MNIST digits).

#include <cstddef>
#include <cstdint>
#include <vector>

// A single fully-connected layer with storage for forward/backward pass data.
struct Layer {
    double* weights;    // Weight matrix [input_size x output_size], row-major
    double* biases;     // Bias vector [output_size]
    double* z;          // Pre-activation values [output_size] (stored for backward pass)
    double* a;          // Post-activation values [output_size] (layer output)
    double* dW;         // Accumulated weight gradients [input_size x output_size]
    double* db;         // Accumulated bias gradients [output_size]
    double* delta;      // Error signal for backprop [output_size]
    size_t  input_size;
    size_t  output_size;
};

// Complete MLP network: a sequence of fully-connected layers.
struct MLP {
    std::vector<Layer> layers;
    size_t num_layers;  // Number of layers (hidden + output, excluding input)
    size_t input_size;  // Network input dimension (e.g., 784 for MNIST)
    size_t output_size; // Network output dimension (e.g., 10 for digit classes)
};

// Create an MLP from a list of layer sizes: {input, hidden1, ..., output}.
// Weights are initialized with He initialization; biases are zeroed.
MLP mlp_create(const std::vector<int>& layer_sizes);

// Run forward pass for a single sample. Returns pointer to output layer
// activations (softmax probabilities). input: [input_size].
const double* mlp_forward(MLP& net, const double* input);

// Run backward pass (backpropagation) for a single sample.
// Computes and accumulates gradients for all layers.
void mlp_backward(MLP& net, const double* input, uint8_t label);

// Apply SGD weight update using accumulated gradients.
void mlp_update(MLP& net, double learning_rate);

// Reset all gradient accumulators (dW, db) to zero.
void mlp_zero_gradients(MLP& net);

// Add gradients from source network into this network's accumulators.
// Used in parallel training to merge per-thread/per-rank gradients.
void mlp_accumulate_gradients(MLP& net, const MLP& source);

// Deallocate all dynamically allocated memory in the network.
void mlp_free(MLP& net);

#endif // MLP_H
