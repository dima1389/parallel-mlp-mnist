#include "optimizer.h"

// Apply vanilla SGD: move each weight in the negative gradient direction.
void sgd_update(double* weights, const double* gradients,
                size_t size, double learning_rate) {
    for (size_t i = 0; i < size; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}
