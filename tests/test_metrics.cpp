#include "metrics.h"
#include "types.h"
#include <cassert>
#include <cmath>
#include <cstdio>

// Unit tests for classification metrics.

static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

static void test_accuracy_perfect() {
    // 3 samples, 3 classes: each prediction has max at the correct class
    real_t preds[] = {
        0.9, 0.05, 0.05,  // predicted class 0
        0.1, 0.8,  0.1,   // predicted class 1
        0.05, 0.05, 0.9   // predicted class 2
    };
    uint8_t labels[] = {0, 1, 2};

    double acc = compute_accuracy(preds, labels, 3, 3);
    assert(approx_eq(acc, 1.0));
    std::printf("  PASS: compute_accuracy (perfect)\n");
}

static void test_accuracy_all_wrong() {
    // 2 samples, 3 classes: all predictions are wrong
    real_t preds[] = {
        0.1, 0.8, 0.1,   // predicted class 1, true class 0
        0.8, 0.1, 0.1    // predicted class 0, true class 2
    };
    uint8_t labels[] = {0, 2};

    double acc = compute_accuracy(preds, labels, 2, 3);
    assert(approx_eq(acc, 0.0));
    std::printf("  PASS: compute_accuracy (all wrong)\n");
}

static void test_accuracy_partial() {
    // 4 samples, 2 classes: 3 out of 4 correct
    real_t preds[] = {
        0.9, 0.1,  // correct (class 0)
        0.3, 0.7,  // correct (class 1)
        0.6, 0.4,  // correct (class 0)
        0.6, 0.4   // wrong (true class 1)
    };
    uint8_t labels[] = {0, 1, 0, 1};

    double acc = compute_accuracy(preds, labels, 4, 2);
    assert(approx_eq(acc, 0.75));
    std::printf("  PASS: compute_accuracy (partial: 75%%)\n");
}

static void test_accuracy_single_sample() {
    real_t preds[] = {0.1, 0.2, 0.7};
    uint8_t labels[] = {2};

    double acc = compute_accuracy(preds, labels, 1, 3);
    assert(approx_eq(acc, 1.0));
    std::printf("  PASS: compute_accuracy (single sample)\n");
}

int main() {
    std::printf("Running metrics tests...\n");
    test_accuracy_perfect();
    test_accuracy_all_wrong();
    test_accuracy_partial();
    test_accuracy_single_sample();
    std::printf("All metrics tests passed!\n\n");
    return 0;
}
