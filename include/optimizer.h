#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// Stochastic Gradient Descent (SGD) optimizer.
// Applies gradient-based weight updates to network parameters.

#include <cstddef>

// SGD update rule: weights[i] -= learning_rate * gradients[i]
// Modifies weights in-place using the provided gradient vector.
void sgd_update(double* weights, const double* gradients,
                size_t size, double learning_rate);

#endif // OPTIMIZER_H
