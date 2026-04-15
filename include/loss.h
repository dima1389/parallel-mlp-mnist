#ifndef LOSS_H
#define LOSS_H

// Cross-entropy loss computation and gradient for softmax output layer.
// Used to measure prediction quality and drive backpropagation.

#include <cstddef>
#include <cstdint>
#include "types.h"

// Cross-entropy loss for a single sample: -log(predicted[label]).
// predicted: softmax probability vector [num_classes], label: true class index.
real_t cross_entropy_loss(const real_t* predicted, uint8_t label, size_t num_classes);

// Average cross-entropy loss over a batch of samples.
real_t cross_entropy_loss_batch(const real_t* predictions, const uint8_t* labels,
                                size_t num_samples, size_t num_classes);

// Combined softmax + cross-entropy gradient for a single sample.
// gradient[i] = predicted[i] - (i == label ? 1 : 0)
// This simplified form avoids computing the Jacobian of softmax separately.
void cross_entropy_softmax_gradient(const real_t* predicted, uint8_t label,
                                    real_t* gradient, size_t num_classes);

#endif // LOSS_H
