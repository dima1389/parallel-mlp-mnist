#include "utils.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>

// Allocate a zero-initialized vector of doubles.
double* alloc_vector(size_t size) {
    double* v = new double[size]();
    return v;
}

// Allocate a zero-initialized 2D matrix stored as a flat 1D array (row-major).
double* alloc_matrix(size_t rows, size_t cols) {
    double* m = new double[rows * cols]();
    return m;
}

// Free a matrix (or vector) allocated with alloc_matrix/alloc_vector.
void free_matrix(double* matrix) {
    delete[] matrix;
}

// He initialization: draw weights from N(0, sqrt(2/fan_in)).
// This variance scaling prevents vanishing/exploding gradients with ReLU.
// Uses a fixed seed (42) for reproducibility across parallel runs.
void he_init(double* weights, size_t fan_in, size_t fan_out) {
    std::mt19937 gen(42);
    double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
    std::normal_distribution<double> dist(0.0, stddev);

    size_t total = fan_in * fan_out;
    for (size_t i = 0; i < total; ++i) {
        weights[i] = dist(gen);
    }
}

// Fill a buffer with zeros using memset (faster than a loop for large arrays).
void zero_init(double* data, size_t size) {
    std::memset(data, 0, size * sizeof(double));
}

// Remove leading and trailing whitespace from a string.
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// Parse a simple key=value configuration file.
// Supported keys: hidden_layers (comma-separated ints), epochs, learning_rate, batch_size.
// Lines starting with '#' are comments. Unknown keys are silently ignored.
Config load_config(const std::string& filepath) {
    Config cfg;
    cfg.epochs        = 10;
    cfg.learning_rate = 0.01;
    cfg.batch_size    = 1;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;  // Skip malformed lines

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (key == "hidden_layers") {
            // Parse comma-separated list of hidden layer sizes
            std::istringstream ss(val);
            std::string token;
            while (std::getline(ss, token, ',')) {
                cfg.hidden_layers.push_back(std::stoi(trim(token)));
            }
        } else if (key == "epochs") {
            cfg.epochs = std::stoi(val);
        } else if (key == "learning_rate") {
            cfg.learning_rate = std::stod(val);
        } else if (key == "batch_size") {
            cfg.batch_size = std::stoi(val);
        }
    }

    return cfg;
}
