#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// Stochastic Gradient Descent (SGD) optimizer.
// Applies gradient-based weight updates to network parameters.

#include <cstddef>
#include "types.h"

// SGD update rule: weights[i] -= learning_rate * gradients[i]
// Modifies weights in-place using the provided gradient vector.
void sgd_update(real_t* weights, const real_t* gradients,
                size_t size, real_t learning_rate);

#endif // OPTIMIZER_H
