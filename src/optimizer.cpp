#include "optimizer.h"

// Apply vanilla SGD: move each weight in the negative gradient direction.
void sgd_update(real_t* weights, const real_t* gradients,
                size_t size, real_t learning_rate) {
    for (size_t i = 0; i < size; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}
