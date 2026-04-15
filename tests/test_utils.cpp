#include "utils.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <fstream>

// Unit tests for utility functions: allocation, He init, config parsing.

static bool approx_eq(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

static void test_alloc_vector_zeroed() {
    real_t* v = alloc_vector(100);
    for (size_t i = 0; i < 100; ++i) {
        assert(v[i] == 0.0);
    }
    free_array(v);
    std::printf("  PASS: alloc_vector (zero-initialized)\n");
}

static void test_alloc_matrix_zeroed() {
    real_t* m = alloc_matrix(10, 20);
    for (size_t i = 0; i < 200; ++i) {
        assert(m[i] == 0.0);
    }
    free_array(m);
    std::printf("  PASS: alloc_matrix (zero-initialized)\n");
}

static void test_zero_init() {
    real_t data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    zero_init(data, 5);
    for (int i = 0; i < 5; ++i) {
        assert(data[i] == 0.0);
    }
    std::printf("  PASS: zero_init\n");
}

static void test_he_init_statistics() {
    // He init for fan_in=100 should produce stddev ≈ sqrt(2/100) ≈ 0.1414
    size_t fan_in = 100, fan_out = 50;
    real_t* w = alloc_matrix(fan_in, fan_out);
    std::mt19937 gen(42);
    he_init(w, fan_in, fan_out, gen);

    double expected_stddev = std::sqrt(2.0 / static_cast<double>(fan_in));

    // Compute sample mean and stddev
    size_t n = fan_in * fan_out;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += w[i];
    double mean = sum / static_cast<double>(n);

    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = w[i] - mean;
        var += d * d;
    }
    double stddev = std::sqrt(var / static_cast<double>(n));

    // Mean should be close to 0, stddev close to expected
    assert(std::fabs(mean) < 0.05);
    assert(std::fabs(stddev - expected_stddev) < 0.03);

    free_array(w);
    std::printf("  PASS: he_init (statistics)\n");
}

static void test_he_init_different_layers() {
    // Two calls to he_init with the same RNG should produce different weights
    // (since the RNG state advances)
    std::mt19937 gen(42);
    real_t w1[10], w2[10];
    he_init(w1, 10, 1, gen);
    he_init(w2, 10, 1, gen);

    bool all_same = true;
    for (int i = 0; i < 10; ++i) {
        if (!approx_eq(w1[i], w2[i])) { all_same = false; break; }
    }
    assert(!all_same);  // Different layers should have different weights
    std::printf("  PASS: he_init (different layers get different weights)\n");
}

static void test_load_config_valid() {
    // Write a temp config file
    const char* path = "test_config_tmp.conf";
    {
        std::ofstream f(path);
        f << "# test config\n";
        f << "hidden_layers = 128, 64\n";
        f << "epochs = 5\n";
        f << "learning_rate = 0.001\n";
        f << "batch_size = 32\n";
    }

    Config cfg = load_config(path);
    assert(cfg.hidden_layers.size() == 2);
    assert(cfg.hidden_layers[0] == 128);
    assert(cfg.hidden_layers[1] == 64);
    assert(cfg.epochs == 5);
    assert(approx_eq(cfg.learning_rate, 0.001));
    assert(cfg.batch_size == 32);

    std::remove(path);
    std::printf("  PASS: load_config (valid file)\n");
}

static void test_load_config_missing_file() {
    bool threw = false;
    try {
        load_config("nonexistent_file.conf");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    std::printf("  PASS: load_config (missing file throws)\n");
}

static void test_load_config_invalid_epochs() {
    const char* path = "test_config_bad_tmp.conf";
    {
        std::ofstream f(path);
        f << "hidden_layers = 128\n";
        f << "epochs = -1\n";
        f << "learning_rate = 0.01\n";
        f << "batch_size = 64\n";
    }

    bool threw = false;
    try {
        load_config(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::remove(path);
    std::printf("  PASS: load_config (invalid epochs throws)\n");
}

int main() {
    std::printf("Running utils tests...\n");
    test_alloc_vector_zeroed();
    test_alloc_matrix_zeroed();
    test_zero_init();
    test_he_init_statistics();
    test_he_init_different_layers();
    test_load_config_valid();
    test_load_config_missing_file();
    test_load_config_invalid_epochs();
    std::printf("All utils tests passed!\n\n");
    return 0;
}
