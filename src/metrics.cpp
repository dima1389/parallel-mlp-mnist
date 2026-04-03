#include "metrics.h"

// Compute classification accuracy as the fraction of correct predictions.
// For each sample, finds the class with highest probability (argmax)
// and checks if it matches the true label.
double compute_accuracy(const double* predictions, const uint8_t* labels,
                        size_t num_samples, size_t num_classes) {
    size_t correct = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        const double* pred = predictions + i * num_classes;

        // Find argmax: the predicted class with highest probability
        size_t max_idx = 0;
        double max_val = pred[0];
        for (size_t j = 1; j < num_classes; ++j) {
            if (pred[j] > max_val) {
                max_val = pred[j];
                max_idx = j;
            }
        }
        if (max_idx == static_cast<size_t>(labels[i])) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}
