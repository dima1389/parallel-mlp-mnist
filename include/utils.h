#ifndef UTILS_H
#define UTILS_H

// Utility functions: memory allocation, weight initialization, and config parsing.

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <map>

// --- Memory allocation helpers (zero-initialized) ---
double* alloc_vector(size_t size);
double* alloc_matrix(size_t rows, size_t cols);
void    free_matrix(double* matrix);

// --- Weight initialization ---
// He initialization: N(0, sqrt(2/fan_in)). Recommended for ReLU networks.
void he_init(double* weights, size_t fan_in, size_t fan_out);

// Set all elements to zero using memset.
void zero_init(double* data, size_t size);

// --- Configuration file parsing ---
// Training hyperparameters loaded from a key=value config file.
struct Config {
    std::vector<int> hidden_layers;  // Hidden layer sizes (e.g., {256, 128})
    int    epochs;                   // Number of training epochs
    double learning_rate;            // SGD learning rate
    int    batch_size;               // Mini-batch size (1 = online SGD)
};

// Parse a config file with keys: hidden_layers, epochs, learning_rate, batch_size.
// Lines starting with '#' are treated as comments.
Config load_config(const std::string& filepath);

// --- Logging helpers (dual stdout + file output) ---
// Global log file pointer. When non-null, log_printf writes to both stdout and this file.
extern FILE* g_log_fp;

// Open a log file at the given path (creates parent directories if needed).
// Subsequent log_printf calls will write to both stdout and this file.
void open_log(const std::string& path);

// Close the log file opened by open_log.
void close_log();

// Printf-style function that writes to stdout and, if a log file is open, also to the log file.
void log_printf(const char* fmt, ...);

#endif // UTILS_H
