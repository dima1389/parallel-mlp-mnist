#include "mlp.h"
#include "activations.h"
#include "loss.h"
#include "optimizer.h"
#include "utils.h"
#include <cstring>
#include <random>

// Construct an MLP from a list of layer sizes (e.g., {784, 256, 128, 10}).
// Allocates weight matrices, bias vectors, and gradient buffers for each layer.
// Uses a single RNG with fixed seed 42 for reproducible He initialization.
MLP mlp_create(const std::vector<int>& layer_sizes) {
    MLP net;
    net.num_layers = layer_sizes.size() - 1;  // Layers between nodes = size - 1
    net.input_size  = static_cast<size_t>(layer_sizes.front());
    net.output_size = static_cast<size_t>(layer_sizes.back());
    net.layers.resize(net.num_layers);

    // Single RNG shared across all layers for unique weight sequences
    std::mt19937 gen(42);

    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l   = net.layers[i];
        l.input_size  = static_cast<size_t>(layer_sizes[i]);
        l.output_size = static_cast<size_t>(layer_sizes[i + 1]);

        // Allocate forward pass buffers
        l.weights = alloc_matrix(l.input_size, l.output_size);
        l.biases  = alloc_vector(l.output_size);
        l.z       = alloc_vector(l.output_size);  // Pre-activation (W*x + b)
        l.a       = alloc_vector(l.output_size);  // Post-activation (f(z))

        // Allocate backward pass buffers
        l.dW      = alloc_matrix(l.input_size, l.output_size);  // Weight gradients
        l.db      = alloc_vector(l.output_size);                // Bias gradients
        l.delta   = alloc_vector(l.output_size);                // Error signal
        l.relu_d  = alloc_vector(l.output_size);                // ReLU derivative buffer

        // Initialize weights with He initialization, biases to zero
        he_init(l.weights, l.input_size, l.output_size, gen);
        zero_init(l.biases, l.output_size);
    }

    return net;
}

// Forward pass: propagate input through all layers.
// Hidden layers use ReLU; the output layer uses Softmax.
// Returns pointer to the output layer's activation (probability vector).
const real_t* mlp_forward(MLP& net, const real_t* input) {
    const real_t* prev_activation = input;

    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l = net.layers[i];

        // Compute pre-activation: z = W^T * prev_activation + b
        // (row-major storage: W[k][j] = weights[k * output_size + j])
        // Loop order k-outer/j-inner for sequential cache-friendly access to weights.
        for (size_t j = 0; j < l.output_size; ++j) {
            l.z[j] = l.biases[j];
        }
        for (size_t k = 0; k < l.input_size; ++k) {
            real_t a_k = prev_activation[k];
            for (size_t j = 0; j < l.output_size; ++j) {
                l.z[j] += a_k * l.weights[k * l.output_size + j];
            }
        }

        // Apply activation function
        if (i < net.num_layers - 1) {
            relu(l.z, l.a, l.output_size);     // Hidden layers: ReLU
        } else {
            softmax(l.z, l.a, l.output_size);  // Output layer: Softmax
        }

        prev_activation = l.a;
    }

    return net.layers.back().a;
}

// Backward pass (backpropagation): compute gradients for all layers.
// Must be called after mlp_forward() so that z and a buffers are populated.
void mlp_backward(MLP& net, const real_t* input, uint8_t label) {
    // --- Step 1: Output layer delta ---
    // For softmax + cross-entropy, the gradient simplifies to: predicted - one_hot
    Layer& out = net.layers[net.num_layers - 1];
    cross_entropy_softmax_gradient(out.a, label, out.delta, out.output_size);

    // --- Step 2: Propagate error signal backward through hidden layers ---
    for (int i = static_cast<int>(net.num_layers) - 2; i >= 0; --i) {
        Layer& curr = net.layers[static_cast<size_t>(i)];
        Layer& next = net.layers[static_cast<size_t>(i) + 1];

        // delta_curr = (W_next^T * delta_next) element-wise-multiply relu'(z_curr)
        relu_derivative(curr.z, curr.relu_d, curr.output_size);

        for (size_t j = 0; j < curr.output_size; ++j) {
            // Sum the weighted error signals from the next layer
            real_t sum = 0.0;
            for (size_t k = 0; k < next.output_size; ++k) {
                sum += next.weights[j * next.output_size + k] * next.delta[k];
            }
            curr.delta[j] = sum * curr.relu_d[j];  // Gate by ReLU derivative
        }
    }

    // --- Step 3: Accumulate weight and bias gradients for each layer ---
    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l = net.layers[i];
        const real_t* prev_a = (i == 0) ? input : net.layers[i - 1].a;

        // dW += outer_product(prev_activation, delta), one sample contribution
        for (size_t j = 0; j < l.input_size; ++j) {
            for (size_t k = 0; k < l.output_size; ++k) {
                l.dW[j * l.output_size + k] += prev_a[j] * l.delta[k];
            }
        }

        // db += delta
        for (size_t k = 0; k < l.output_size; ++k) {
            l.db[k] += l.delta[k];
        }
    }
}

// Apply SGD update to all weights and biases using accumulated gradients.
void mlp_update(MLP& net, real_t learning_rate) {
    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l = net.layers[i];
        sgd_update(l.weights, l.dW, l.input_size * l.output_size, learning_rate);
        sgd_update(l.biases, l.db, l.output_size, learning_rate);
    }
}

// Reset all gradient accumulators to zero before processing a new batch.
void mlp_zero_gradients(MLP& net) {
    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l = net.layers[i];
        zero_init(l.dW, l.input_size * l.output_size);
        zero_init(l.db, l.output_size);
    }
}

// Merge gradients from a source network into this network's accumulators.
// Used in parallel training: each thread/rank accumulates locally, then
// all gradients are summed into the master network before the update step.
void mlp_accumulate_gradients(MLP& net, const MLP& source) {
    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& dst = net.layers[i];
        const Layer& src = source.layers[i];
        size_t w_size = dst.input_size * dst.output_size;
        for (size_t j = 0; j < w_size; ++j) {
            dst.dW[j] += src.dW[j];
        }
        for (size_t j = 0; j < dst.output_size; ++j) {
            dst.db[j] += src.db[j];
        }
    }
}

// Count total trainable parameters (weights + biases) across all layers.
size_t mlp_total_params(const MLP& net) {
    size_t total = 0;
    for (size_t i = 0; i < net.num_layers; ++i) {
        total += net.layers[i].input_size * net.layers[i].output_size; // weights
        total += net.layers[i].output_size;                           // biases
    }
    return total;
}

// Deallocate all dynamically allocated arrays in every layer.
void mlp_free(MLP& net) {
    for (size_t i = 0; i < net.num_layers; ++i) {
        Layer& l = net.layers[i];
        free_array(l.weights);
        free_array(l.biases);
        free_array(l.z);
        free_array(l.a);
        free_array(l.dW);
        free_array(l.db);
        free_array(l.delta);
        free_array(l.relu_d);
    }
    net.layers.clear();
    net.num_layers = 0;
}
