// Sequential (single-threaded) MLP training on MNIST.
// Implements mini-batch SGD: accumulates gradients over a batch, then updates.

#include "dataset.h"
#include "mlp.h"
#include "loss.h"
#include "metrics.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s --config <config_file> [--data <mnist_dir>] [--log <log_file>] [--epoch-csv <csv_file>]\n", prog);
}

int main(int argc, char* argv[]) {
    std::string config_path;
    std::string data_dir = "data/mnist/raw";
    std::string log_path;
    std::string epoch_csv_path;

    // Parse command-line arguments: --config, --data, --log, --epoch-csv
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "--epoch-csv" && i + 1 < argc) {
            epoch_csv_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load training hyperparameters from config file
    Config cfg = load_config(config_path);

    if (!log_path.empty()) open_log(log_path);

    log_printf("=== Sequential MLP Training ===\n");
    log_printf("Hidden layers: ");
    for (size_t i = 0; i < cfg.hidden_layers.size(); ++i) {
        log_printf("%d%s", cfg.hidden_layers[i],
                    (i < cfg.hidden_layers.size() - 1) ? ", " : "");
    }
    log_printf("\nEpochs: %d, LR: %.4f\n\n", cfg.epochs, cfg.learning_rate);

    // Load MNIST training and test splits
    log_printf("Loading MNIST data from %s ...\n", data_dir.c_str());
    Dataset train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    Dataset train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    Dataset test_images  = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    Dataset test_labels  = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    log_printf("Train: %zu samples, Test: %zu samples, Image size: %zu\n",
                train_images.num_samples, test_images.num_samples, train_images.image_size);

    // Build network architecture: [784, hidden1, ..., hiddenN, 10]
    std::vector<int> layer_sizes;
    layer_sizes.push_back(static_cast<int>(train_images.image_size));
    for (int h : cfg.hidden_layers) {
        layer_sizes.push_back(h);
    }
    layer_sizes.push_back(10); // 10 digit classes (0-9)

    // Initialize the MLP with He-initialized weights
    MLP net = mlp_create(layer_sizes);
    size_t total_params = mlp_total_params(net);

    // Buffer to store predictions for test set accuracy evaluation
    size_t num_classes = 10;
    real_t* test_preds = alloc_matrix(test_images.num_samples, num_classes);

    Timer epoch_timer, total_timer;
    std::vector<EpochRecord> epoch_records;
    double cumulative_time = 0.0;

    // --- Training loop: mini-batch SGD ---
    timer_start(total_timer);

    size_t N = train_images.num_samples;
    size_t batch_size = static_cast<size_t>(cfg.batch_size);

    // Index array for epoch-level shuffling
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 shuffle_rng(42);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        timer_start(epoch_timer);

        // Shuffle sample order each epoch for better convergence
        std::shuffle(indices.begin(), indices.end(), shuffle_rng);

        double epoch_loss = 0.0;

        // Process training samples in mini-batches
        for (size_t start = 0; start < N; start += batch_size) {
            size_t end = std::min(start + batch_size, N);
            size_t actual_batch = end - start;

            mlp_zero_gradients(net);

            for (size_t si = start; si < end; ++si) {
                size_t s = indices[si];
                const real_t* img = train_images.images + s * train_images.image_size;
                uint8_t label = train_labels.labels[s];

                // Forward pass: compute predictions
                const real_t* output = mlp_forward(net, img);

                // Track running loss for this epoch
                epoch_loss += cross_entropy_loss(output, label, num_classes);

                // Backward pass: accumulate gradients
                mlp_backward(net, img, label);
            }

            // Average gradients over the mini-batch, then update weights
            double inv_batch = 1.0 / static_cast<double>(actual_batch);
            for (size_t l = 0; l < net.num_layers; ++l) {
                Layer& layer = net.layers[l];
                size_t w_size = layer.input_size * layer.output_size;
                for (size_t j = 0; j < w_size; ++j) {
                    layer.dW[j] *= inv_batch;
                }
                for (size_t j = 0; j < layer.output_size; ++j) {
                    layer.db[j] *= inv_batch;
                }
            }
            mlp_update(net, cfg.learning_rate);
        }

        timer_stop(epoch_timer);
        epoch_loss /= static_cast<double>(N);

        // Evaluate accuracy on the full test set
        for (size_t s = 0; s < test_images.num_samples; ++s) {
            const real_t* img = test_images.images + s * test_images.image_size;
            const real_t* output = mlp_forward(net, img);
            for (size_t c = 0; c < num_classes; ++c) {
                test_preds[s * num_classes + c] = output[c];
            }
        }
        double accuracy = compute_accuracy(test_preds, test_labels.labels,
                                           test_images.num_samples, num_classes);

        log_printf("Epoch %2d/%d  Loss: %.4f  Accuracy: %.2f%%  Time: %.3f s\n",
                    epoch + 1, cfg.epochs, epoch_loss, accuracy * 100.0,
                    timer_elapsed_sec(epoch_timer));

        cumulative_time += timer_elapsed_sec(epoch_timer);
        epoch_records.push_back({epoch + 1, epoch_loss, accuracy,
                                 timer_elapsed_sec(epoch_timer), cumulative_time});
    }

    timer_stop(total_timer);

    if (!epoch_csv_path.empty()) {
        write_epoch_csv(epoch_csv_path, epoch_records);
    }
    print_training_summary(total_params, N, timer_elapsed_sec(total_timer), epoch_records);

    close_log();

    // Cleanup all allocated memory
    free_array(test_preds);
    mlp_free(net);
    free_dataset(train_images);
    free_dataset(train_labels);
    free_dataset(test_images);
    free_dataset(test_labels);

    return 0;
}
