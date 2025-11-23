#pragma once

#include "../tensor.hpp"

/**
 * ReLU activation layer (CPU implementation)
 */
class ReLULayer {
public:
    Tensor forward(const Tensor& input);
};

