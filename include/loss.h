#ifndef LOSS_H
#define LOSS_H

// Cross-entropy loss computation and gradient for softmax output layer.
// Used to measure prediction quality and drive backpropagation.

#include <cstddef>
#include <cstdint>

// Cross-entropy loss for a single sample: -log(predicted[label]).
// predicted: softmax probability vector [num_classes], label: true class index.
double cross_entropy_loss(const double* predicted, uint8_t label, size_t num_classes);

// Average cross-entropy loss over a batch of samples.
double cross_entropy_loss_batch(const double* predictions, const uint8_t* labels,
                                size_t num_samples, size_t num_classes);

// Combined softmax + cross-entropy gradient for a single sample.
// gradient[i] = predicted[i] - (i == label ? 1 : 0)
// This simplified form avoids computing the Jacobian of softmax separately.
void cross_entropy_softmax_gradient(const double* predicted, uint8_t label,
                                    double* gradient, size_t num_classes);

#endif // LOSS_H
