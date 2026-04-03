#include "mlp.h"
#include "utils.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Unit tests for the MLP network: creation, forward pass, backward pass, and training.

// Helper: check if two doubles are approximately equal within tolerance.
static bool approx_eq(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

// Test network creation and verify layer dimensions are set correctly.
static void test_mlp_create_free() {
    std::vector<int> sizes = {4, 3, 2};  // 4 inputs -> 3 hidden -> 2 outputs
    MLP net = mlp_create(sizes);

    assert(net.num_layers == 2);           // 2 layers (hidden + output)
    assert(net.input_size == 4);
    assert(net.output_size == 2);
    assert(net.layers[0].input_size == 4);  // First layer: 4->3
    assert(net.layers[0].output_size == 3);
    assert(net.layers[1].input_size == 3);  // Second layer: 3->2
    assert(net.layers[1].output_size == 2);

    mlp_free(net);
    assert(net.num_layers == 0);           // Verify cleanup
    std::printf("  PASS: mlp_create / mlp_free\n");
}

// Verify forward pass produces valid softmax output (sums to 1, all in [0,1]).
static void test_forward_output_shape() {
    std::vector<int> sizes = {4, 3, 2};
    MLP net = mlp_create(sizes);

    double input[] = {0.1, 0.2, 0.3, 0.4};
    const double* output = mlp_forward(net, input);

    // Softmax output must sum to 1.0
    double sum = output[0] + output[1];
    assert(approx_eq(sum, 1.0));

    // Each probability must be in [0, 1]
    assert(output[0] >= 0.0 && output[0] <= 1.0);
    assert(output[1] >= 0.0 && output[1] <= 1.0);

    mlp_free(net);
    std::printf("  PASS: forward output is valid softmax\n");
}

// Verify that backward pass produces non-zero gradients (learning is possible).
static void test_backward_updates_gradients() {
    std::vector<int> sizes = {4, 3, 2};
    MLP net = mlp_create(sizes);

    double input[] = {0.1, 0.2, 0.3, 0.4};
    mlp_zero_gradients(net);
    mlp_forward(net, input);
    mlp_backward(net, input, 0);

    // At least one weight gradient should be non-zero after backward pass
    bool has_nonzero = false;
    for (size_t l = 0; l < net.num_layers; ++l) {
        size_t w_size = net.layers[l].input_size * net.layers[l].output_size;
        for (size_t j = 0; j < w_size; ++j) {
            if (net.layers[l].dW[j] != 0.0) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) break;
    }
    assert(has_nonzero);

    mlp_free(net);
    std::printf("  PASS: backward produces non-zero gradients\n");
}

// End-to-end test: repeated training on a fixed sample should reduce the loss.
static void test_training_reduces_loss() {
    std::vector<int> sizes = {4, 8, 2};
    MLP net = mlp_create(sizes);

    double input[] = {0.5, 0.3, 0.8, 0.1};
    uint8_t label = 1;
    double lr = 0.1;

    // Measure initial loss
    const double* out = mlp_forward(net, input);
    double initial_loss = -std::log(std::max(out[label], 1e-12));

    // Train for 50 SGD steps on the same sample
    for (int i = 0; i < 50; ++i) {
        mlp_zero_gradients(net);
        mlp_forward(net, input);
        mlp_backward(net, input, label);
        mlp_update(net, lr);
    }

    // Loss should have decreased
    out = mlp_forward(net, input);
    double final_loss = -std::log(std::max(out[label], 1e-12));
    assert(final_loss < initial_loss);

    mlp_free(net);
    std::printf("  PASS: training reduces loss (%.4f -> %.4f)\n", initial_loss, final_loss);
}

int main() {
    std::printf("=== test_mlp ===\n");
    test_mlp_create_free();
    test_forward_output_shape();
    test_backward_updates_gradients();
    test_training_reduces_loss();
    std::printf("All MLP tests passed.\n");
    return 0;
}
