#include "activations.h"
#include "loss.h"
#include "types.h"
#include <cassert>
#include <cmath>
#include <cstdio>

// Unit tests for activation functions and cross-entropy loss.

// Helper: check if two doubles are approximately equal within tolerance.
static bool approx_eq(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

// Test that ReLU zeroes negatives and passes through positives.
static void test_relu() {
    real_t input[]  = {-2.0, -0.5, 0.0, 0.5, 2.0};
    real_t output[5];
    relu(input, output, 5);

    assert(output[0] == 0.0);
    assert(output[1] == 0.0);
    assert(output[2] == 0.0);
    assert(output[3] == 0.5);
    assert(output[4] == 2.0);
    std::printf("  PASS: relu\n");
}

// Test that ReLU derivative is 1 for positive, 0 for non-positive.
static void test_relu_derivative() {
    real_t input[]  = {-2.0, -0.5, 0.0, 0.5, 2.0};
    real_t output[5];
    relu_derivative(input, output, 5);

    assert(output[0] == 0.0);
    assert(output[1] == 0.0);
    assert(output[2] == 0.0);
    assert(output[3] == 1.0);
    assert(output[4] == 1.0);
    std::printf("  PASS: relu_derivative\n");
}

// Test that softmax outputs sum to 1 and preserve input ordering.
static void test_softmax() {
    real_t input[]  = {1.0, 2.0, 3.0};
    real_t output[3];
    softmax(input, output, 3);

    // Probabilities must sum to 1.0
    double sum = output[0] + output[1] + output[2];
    assert(approx_eq(sum, 1.0));

    // Larger input should map to larger probability
    assert(output[0] < output[1]);
    assert(output[1] < output[2]);
    std::printf("  PASS: softmax\n");
}

// Test that softmax handles very large inputs without overflow.
static void test_softmax_numerically_stable() {
    real_t input[]  = {1000.0, 1001.0, 1002.0};
    real_t output[3];
    softmax(input, output, 3);

    double sum = output[0] + output[1] + output[2];
    assert(approx_eq(sum, 1.0));
    std::printf("  PASS: softmax (large values)\n");
}

// Test CE loss: correct prediction should give low loss, wrong one high loss.
static void test_cross_entropy_loss() {
    // Good prediction: high probability on correct class -> low loss
    real_t pred_perfect[] = {0.01, 0.01, 0.96, 0.01, 0.01};
    real_t loss = cross_entropy_loss(pred_perfect, 2, 5);
    assert(loss < 0.1);
    std::printf("  PASS: cross_entropy_loss (good prediction)\n");

    // Bad prediction: low probability on correct class -> high loss
    real_t pred_bad[] = {0.01, 0.96, 0.01, 0.01, 0.01};
    loss = cross_entropy_loss(pred_bad, 2, 5);
    assert(loss > 2.0);
    std::printf("  PASS: cross_entropy_loss (bad prediction)\n");
}

// Test gradient: grad[label] = pred - 1, grad[other] = pred.
static void test_cross_entropy_gradient() {
    real_t pred[] = {0.1, 0.2, 0.5, 0.1, 0.1};
    real_t grad[5];
    cross_entropy_softmax_gradient(pred, 2, grad, 5);

    assert(approx_eq(grad[0], 0.1));
    assert(approx_eq(grad[1], 0.2));
    assert(approx_eq(grad[2], -0.5));   // 0.5 - 1.0 = -0.5
    assert(approx_eq(grad[3], 0.1));
    assert(approx_eq(grad[4], 0.1));
    std::printf("  PASS: cross_entropy_softmax_gradient\n");
}

int main() {
    std::printf("=== test_loss ===\n");
    test_relu();
    test_relu_derivative();
    test_softmax();
    test_softmax_numerically_stable();
    test_cross_entropy_loss();
    test_cross_entropy_gradient();
    std::printf("All loss/activation tests passed.\n");
    return 0;
}
