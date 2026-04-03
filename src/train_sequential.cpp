// Sequential (single-threaded) MLP training on MNIST.
// Implements online SGD: updates weights after every sample.

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

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s --config <config_file> --data <mnist_dir>\n", prog);
}

int main(int argc, char* argv[]) {
    std::string config_path;
    std::string data_dir = "data/mnist/raw";

    // Parse command-line arguments: --config and --data
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
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

    std::printf("=== Sequential MLP Training ===\n");
    std::printf("Hidden layers: ");
    for (size_t i = 0; i < cfg.hidden_layers.size(); ++i) {
        std::printf("%d%s", cfg.hidden_layers[i],
                    (i < cfg.hidden_layers.size() - 1) ? ", " : "");
    }
    std::printf("\nEpochs: %d, LR: %.4f\n\n", cfg.epochs, cfg.learning_rate);

    // Load MNIST training and test splits
    std::printf("Loading MNIST data from %s ...\n", data_dir.c_str());
    Dataset train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    Dataset train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    Dataset test_images  = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    Dataset test_labels  = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    std::printf("Train: %zu samples, Test: %zu samples, Image size: %zu\n",
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

    // Buffer to store predictions for test set accuracy evaluation
    size_t num_classes = 10;
    double* test_preds = alloc_matrix(test_images.num_samples, num_classes);

    Timer epoch_timer, total_timer;

    // --- Training loop: one pass over all training samples per epoch ---
    timer_start(total_timer);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        timer_start(epoch_timer);

        double epoch_loss = 0.0;

        // Process each training sample (online SGD)
        for (size_t s = 0; s < train_images.num_samples; ++s) {
            const double* img = train_images.images + s * train_images.image_size;
            uint8_t label = train_labels.labels[s];

            // Forward pass: compute predictions
            mlp_zero_gradients(net);
            const double* output = mlp_forward(net, img);

            // Track running loss for this epoch
            epoch_loss += cross_entropy_loss(output, label, num_classes);

            // Backward pass: compute gradients
            mlp_backward(net, img, label);

            // Online SGD: update weights immediately after each sample
            mlp_update(net, cfg.learning_rate);
        }

        timer_stop(epoch_timer);
        epoch_loss /= static_cast<double>(train_images.num_samples);

        // Evaluate accuracy on the full test set
        for (size_t s = 0; s < test_images.num_samples; ++s) {
            const double* img = test_images.images + s * test_images.image_size;
            const double* output = mlp_forward(net, img);
            for (size_t c = 0; c < num_classes; ++c) {
                test_preds[s * num_classes + c] = output[c];
            }
        }
        double accuracy = compute_accuracy(test_preds, test_labels.labels,
                                           test_images.num_samples, num_classes);

        std::printf("Epoch %2d/%d  Loss: %.4f  Accuracy: %.2f%%  Time: %.3f s\n",
                    epoch + 1, cfg.epochs, epoch_loss, accuracy * 100.0,
                    timer_elapsed_sec(epoch_timer));
    }

    timer_stop(total_timer);
    std::printf("\nTotal training time: %.3f s\n", timer_elapsed_sec(total_timer));

    // Cleanup all allocated memory
    free_matrix(test_preds);
    mlp_free(net);
    free_dataset(train_images);
    free_dataset(train_labels);
    free_dataset(test_images);
    free_dataset(test_labels);

    return 0;
}
