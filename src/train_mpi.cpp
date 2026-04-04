// MPI-parallelized MLP training on MNIST.
// Uses data parallelism across MPI ranks: each rank processes a shard of
// the training data, then gradients are summed via MPI_Allreduce and averaged.

#include "dataset.h"
#include "mlp.h"
#include "loss.h"
#include "metrics.h"
#include "timer.h"
#include "utils.h"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: mpirun -np <N> %s --config <config_file> --data <mnist_dir>\n", prog);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // This process's ID
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Total number of processes

    std::string config_path;
    std::string data_dir = "data/mnist/raw";

    // Parse command-line arguments: --config and --data
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (rank == 0) {
            print_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (config_path.empty()) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    // All ranks load the config file (small enough to avoid scatter overhead)
    Config cfg = load_config(config_path);

    if (rank == 0) {
        std::printf("=== MPI MLP Training (%d processes) ===\n", world_size);
        std::printf("Hidden layers: ");
        for (size_t i = 0; i < cfg.hidden_layers.size(); ++i) {
            std::printf("%d%s", cfg.hidden_layers[i],
                        (i < cfg.hidden_layers.size() - 1) ? ", " : "");
        }
        std::printf("\nEpochs: %d, LR: %.4f\n\n", cfg.epochs, cfg.learning_rate);
    }

    // All ranks load full dataset (simpler than MPI scatter for IDX format)
    Dataset train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    Dataset train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    Dataset test_images  = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    Dataset test_labels  = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (rank == 0) {
        std::printf("Train: %zu samples, Test: %zu samples\n",
                    train_images.num_samples, test_images.num_samples);
    }

    size_t N = train_images.num_samples;
    size_t batch_size = static_cast<size_t>(cfg.batch_size);

    // Build network architecture
    std::vector<int> layer_sizes;
    layer_sizes.push_back(static_cast<int>(train_images.image_size));
    for (int h : cfg.hidden_layers) {
        layer_sizes.push_back(h);
    }
    layer_sizes.push_back(10);

    size_t num_classes = 10;

    // All ranks create the network with the same seed, ensuring identical initial weights
    MLP net = mlp_create(layer_sizes);

    // Count total parameters (weights + biases) for the Allreduce buffer
    size_t total_params = 0;
    for (size_t l = 0; l < net.num_layers; ++l) {
        total_params += net.layers[l].input_size * net.layers[l].output_size; // weights
        total_params += net.layers[l].output_size;                           // biases
    }

    // Flat buffers for packing gradients before Allreduce
    double* local_grad  = alloc_vector(total_params);
    double* global_grad = alloc_vector(total_params);

    // Only rank 0 needs the test prediction buffer
    double* test_preds = nullptr;
    if (rank == 0) {
        test_preds = alloc_matrix(test_images.num_samples, num_classes);
    }

    Timer epoch_timer, total_timer;

    if (rank == 0) timer_start(total_timer);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        if (rank == 0) timer_start(epoch_timer);

        double local_loss = 0.0;

        // --- Mini-batch parallel SGD across MPI ranks ---
        // Process `batch_size` samples at a time, divided among ranks.
        // Gradients are accumulated locally, then summed via one Allreduce per batch.
        for (size_t batch_start = 0; batch_start < N; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, N);
            size_t actual_batch = batch_end - batch_start;

            // Divide the batch among ranks
            size_t local_bs = actual_batch / static_cast<size_t>(world_size);
            size_t local_rem = actual_batch % static_cast<size_t>(world_size);
            size_t local_offset = batch_start +
                static_cast<size_t>(rank) * local_bs +
                std::min(static_cast<size_t>(rank), local_rem);
            size_t local_count = local_bs + (static_cast<size_t>(rank) < local_rem ? 1 : 0);

            // Each rank accumulates gradients over its local portion of the batch
            mlp_zero_gradients(net);
            for (size_t s = local_offset; s < local_offset + local_count; ++s) {
                const double* img = train_images.images + s * train_images.image_size;
                uint8_t label = train_labels.labels[s];

                const double* output = mlp_forward(net, img);
                local_loss += cross_entropy_loss(output, label, num_classes);
                mlp_backward(net, img, label);
            }

            // Pack accumulated gradients into flat buffer
            size_t offset = 0;
            for (size_t l = 0; l < net.num_layers; ++l) {
                Layer& layer = net.layers[l];
                size_t w_size = layer.input_size * layer.output_size;
                std::memcpy(local_grad + offset, layer.dW, w_size * sizeof(double));
                offset += w_size;
                std::memcpy(local_grad + offset, layer.db, layer.output_size * sizeof(double));
                offset += layer.output_size;
            }

            // One Allreduce per mini-batch: sum gradients across all ranks
            MPI_Allreduce(local_grad, global_grad, static_cast<int>(total_params),
                           MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Unpack, average over actual_batch size, and apply SGD update
            double inv_batch = 1.0 / static_cast<double>(actual_batch);
            offset = 0;
            for (size_t l = 0; l < net.num_layers; ++l) {
                Layer& layer = net.layers[l];
                size_t w_size = layer.input_size * layer.output_size;
                for (size_t j = 0; j < w_size; ++j) {
                    layer.dW[j] = global_grad[offset + j] * inv_batch;
                }
                offset += w_size;
                for (size_t j = 0; j < layer.output_size; ++j) {
                    layer.db[j] = global_grad[offset + j] * inv_batch;
                }
                offset += layer.output_size;
            }
            mlp_update(net, cfg.learning_rate);
        }

        // Reduce loss to rank 0 for reporting
        double global_loss = 0.0;
        MPI_Reduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Rank 0 evaluates on test set and prints epoch summary
        if (rank == 0) {
            timer_stop(epoch_timer);
            global_loss /= static_cast<double>(N);

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
                        epoch + 1, cfg.epochs, global_loss, accuracy * 100.0,
                        timer_elapsed_sec(epoch_timer));
        }
    }

    if (rank == 0) {
        timer_stop(total_timer);
        std::printf("\nTotal training time: %.3f s\n", timer_elapsed_sec(total_timer));
        free_matrix(test_preds);
    }

    // Cleanup: all ranks free their resources
    free_matrix(local_grad);
    free_matrix(global_grad);
    mlp_free(net);
    free_dataset(train_images);
    free_dataset(train_labels);
    free_dataset(test_images);
    free_dataset(test_labels);

    MPI_Finalize();
    return 0;
}
