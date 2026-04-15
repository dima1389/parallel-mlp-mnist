#include "optimizer.h"
#include "types.h"
#include <cassert>
#include <cmath>
#include <cstdio>

// Unit tests for the SGD optimizer.

static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

static void test_sgd_update_basic() {
    real_t weights[]   = {1.0, 2.0, 3.0};
    real_t gradients[] = {0.1, 0.2, 0.3};
    real_t lr = 0.5;

    sgd_update(weights, gradients, 3, lr);

    // weights[i] -= lr * gradients[i]
    assert(approx_eq(weights[0], 1.0 - 0.5 * 0.1));
    assert(approx_eq(weights[1], 2.0 - 0.5 * 0.2));
    assert(approx_eq(weights[2], 3.0 - 0.5 * 0.3));
    std::printf("  PASS: sgd_update (basic)\n");
}

static void test_sgd_update_zero_gradient() {
    real_t weights[]   = {5.0, -3.0};
    real_t gradients[] = {0.0, 0.0};

    sgd_update(weights, gradients, 2, 0.1);

    assert(approx_eq(weights[0], 5.0));
    assert(approx_eq(weights[1], -3.0));
    std::printf("  PASS: sgd_update (zero gradient)\n");
}

static void test_sgd_update_zero_lr() {
    real_t weights[]   = {1.0, 2.0};
    real_t gradients[] = {10.0, 20.0};

    sgd_update(weights, gradients, 2, 0.0);

    assert(approx_eq(weights[0], 1.0));
    assert(approx_eq(weights[1], 2.0));
    std::printf("  PASS: sgd_update (zero learning rate)\n");
}

static void test_sgd_update_negative_gradients() {
    real_t weights[]   = {0.0, 0.0};
    real_t gradients[] = {-1.0, -2.0};
    real_t lr = 1.0;

    sgd_update(weights, gradients, 2, lr);

    // Negative gradients should increase weights
    assert(approx_eq(weights[0], 1.0));
    assert(approx_eq(weights[1], 2.0));
    std::printf("  PASS: sgd_update (negative gradients)\n");
}

int main() {
    std::printf("Running optimizer tests...\n");
    test_sgd_update_basic();
    test_sgd_update_zero_gradient();
    test_sgd_update_zero_lr();
    test_sgd_update_negative_gradients();
    std::printf("All optimizer tests passed!\n\n");
    return 0;
}
