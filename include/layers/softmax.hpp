#pragma once

#include "../tensor.hpp"
#include <cmath>

/**
 * Softmax layer (CPU implementation)
 */
class SoftmaxLayer {
public:
    Tensor forward(const Tensor& input);
};

