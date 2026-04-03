#ifndef METRICS_H
#define METRICS_H

// Classification evaluation metrics for the MNIST digit recognition task.

#include <cstddef>
#include <cstdint>

// Compute top-1 classification accuracy: fraction of samples where the
// predicted class (argmax of softmax output) matches the true label.
// predictions: [num_samples x num_classes], labels: [num_samples].
double compute_accuracy(const double* predictions, const uint8_t* labels,
                        size_t num_samples, size_t num_classes);

#endif // METRICS_H
