#include "utils.h"
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// Allocate a zero-initialized vector.
real_t* alloc_vector(size_t size) {
    real_t* v = new real_t[size]();
    return v;
}

// Allocate a zero-initialized 2D matrix stored as a flat 1D array (row-major).
real_t* alloc_matrix(size_t rows, size_t cols) {
    real_t* m = new real_t[rows * cols]();
    return m;
}

// Free an array allocated with alloc_matrix/alloc_vector.
void free_array(real_t* arr) {
    delete[] arr;
}

// He initialization: draw weights from N(0, sqrt(2/fan_in)).
// This variance scaling prevents vanishing/exploding gradients with ReLU.
// Uses an external RNG so that successive layers draw unique sequences.
void he_init(real_t* weights, size_t fan_in, size_t fan_out, std::mt19937& gen) {
    double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
    std::normal_distribution<double> dist(0.0, stddev);

    size_t total = fan_in * fan_out;
    for (size_t i = 0; i < total; ++i) {
        weights[i] = static_cast<real_t>(dist(gen));
    }
}

// Fill a buffer with zeros using memset (faster than a loop for large arrays).
void zero_init(real_t* data, size_t size) {
    std::memset(data, 0, size * sizeof(real_t));
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

    // Validate parsed configuration values
    if (cfg.hidden_layers.empty()) {
        throw std::runtime_error("Config error: hidden_layers must specify at least one layer size");
    }
    if (cfg.epochs <= 0) {
        throw std::runtime_error("Config error: epochs must be positive (got " + std::to_string(cfg.epochs) + ")");
    }
    if (cfg.learning_rate <= 0.0) {
        throw std::runtime_error("Config error: learning_rate must be positive (got " + std::to_string(cfg.learning_rate) + ")");
    }
    if (cfg.batch_size < 1) {
        throw std::runtime_error("Config error: batch_size must be >= 1 (got " + std::to_string(cfg.batch_size) + ")");
    }

    return cfg;
}

// --- Logging helpers ---

FILE* g_log_fp = nullptr;

// Create parent directories for a file path (cross-platform).
static void make_parent_dirs(const std::string& filepath) {
    size_t pos = 0;
    while ((pos = filepath.find_first_of("/\\", pos + 1)) != std::string::npos) {
        std::string dir = filepath.substr(0, pos);
#ifdef _WIN32
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0755);
#endif
    }
}

void open_log(const std::string& path) {
    make_parent_dirs(path);
    g_log_fp = std::fopen(path.c_str(), "w");
    if (!g_log_fp) {
        std::fprintf(stderr, "Warning: could not open log file: %s\n", path.c_str());
    }
}

void close_log() {
    if (g_log_fp) {
        std::fclose(g_log_fp);
        g_log_fp = nullptr;
    }
}

void log_printf(const char* fmt, ...) {
    va_list args;

    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);

    if (g_log_fp) {
        va_start(args, fmt);
        std::vfprintf(g_log_fp, fmt, args);
        va_end(args);
        std::fflush(g_log_fp);
    }
}

// --- Training metrics ---

void write_epoch_csv(const std::string& path, const std::vector<EpochRecord>& records) {
    make_parent_dirs(path);
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr, "Warning: could not open epoch CSV: %s\n", path.c_str());
        return;
    }
    std::fprintf(fp, "epoch,train_loss,test_accuracy,epoch_time_sec,cumulative_time_sec\n");
    for (const auto& r : records) {
        std::fprintf(fp, "%d,%.6f,%.6f,%.4f,%.4f\n",
                     r.epoch, r.train_loss, r.test_accuracy,
                     r.epoch_time_sec, r.cumulative_time_sec);
    }
    std::fclose(fp);
}

void print_training_summary(size_t total_params, size_t train_samples,
                            double total_time_sec,
                            const std::vector<EpochRecord>& records) {
    int    epochs         = static_cast<int>(records.size());
    double final_loss     = records.back().train_loss;
    double final_accuracy = records.back().test_accuracy;

    // Find best accuracy and its epoch
    double best_accuracy      = 0.0;
    int    best_accuracy_epoch = 1;
    for (const auto& r : records) {
        if (r.test_accuracy > best_accuracy) {
            best_accuracy       = r.test_accuracy;
            best_accuracy_epoch = r.epoch;
        }
    }

    // Compute epoch time statistics
    double sum_t  = 0.0;
    double min_t  = records[0].epoch_time_sec;
    double max_t  = records[0].epoch_time_sec;
    for (const auto& r : records) {
        sum_t += r.epoch_time_sec;
        if (r.epoch_time_sec < min_t) min_t = r.epoch_time_sec;
        if (r.epoch_time_sec > max_t) max_t = r.epoch_time_sec;
    }
    double avg_t = sum_t / epochs;

    double var_t = 0.0;
    for (const auto& r : records) {
        double d = r.epoch_time_sec - avg_t;
        var_t += d * d;
    }
    double stddev_t = (epochs > 1) ? std::sqrt(var_t / epochs) : 0.0;

    double throughput = (avg_t > 0.0)
        ? static_cast<double>(train_samples) / avg_t
        : 0.0;

    log_printf("\n[SUMMARY]\n");
    log_printf("total_time=%.4f\n",               total_time_sec);
    log_printf("epochs=%d\n",                      epochs);
    log_printf("final_loss=%.6f\n",                final_loss);
    log_printf("final_accuracy=%.6f\n",            final_accuracy);
    log_printf("best_accuracy=%.6f\n",             best_accuracy);
    log_printf("best_accuracy_epoch=%d\n",         best_accuracy_epoch);
    log_printf("avg_epoch_time=%.4f\n",            avg_t);
    log_printf("min_epoch_time=%.4f\n",            min_t);
    log_printf("max_epoch_time=%.4f\n",            max_t);
    log_printf("stddev_epoch_time=%.4f\n",         stddev_t);
    log_printf("throughput_samples_per_sec=%.2f\n", throughput);
    log_printf("total_params=%zu\n",               total_params);
    log_printf("[/SUMMARY]\n");
}
