#include "loss.h"
#include <cmath>
#include <algorithm>

// Small constant to prevent log(0) which would produce -infinity.
static constexpr real_t EPSILON = 1e-12;

// Compute cross-entropy loss for one sample: -log(P(true_class)).
// Clamps the predicted probability to avoid numerical issues with log(0).
real_t cross_entropy_loss(const real_t* predicted, uint8_t label, size_t /*num_classes*/) {
    real_t p = predicted[label];
    p = std::max(p, EPSILON); // Clamp to avoid log(0)
    return -std::log(p);
}

// Compute mean cross-entropy loss across a batch of samples.
real_t cross_entropy_loss_batch(const real_t* predictions, const uint8_t* labels,
                                size_t num_samples, size_t num_classes) {
    real_t total = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        total += cross_entropy_loss(predictions + i * num_classes, labels[i], num_classes);
    }
    return total / static_cast<real_t>(num_samples);
}

// Compute the combined softmax + cross-entropy gradient.
// This is the derivative of CE loss w.r.t. the pre-softmax logits:
// grad[i] = softmax_output[i] - one_hot[i], which simplifies the math.
void cross_entropy_softmax_gradient(const real_t* predicted, uint8_t label,
                                    real_t* gradient, size_t num_classes) {
    for (size_t i = 0; i < num_classes; ++i) {
        gradient[i] = predicted[i] - (i == static_cast<size_t>(label) ? 1.0 : 0.0);
    }
}
