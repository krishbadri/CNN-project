#pragma once

#include "../tensor.hpp"

/**
 * Max pooling layer (CPU implementation)
 */
class MaxPoolLayer {
public:
    MaxPoolLayer(int pool_size = 2) : pool_size_(pool_size) {}
    
    Tensor forward(const Tensor& input);

private:
    int pool_size_;
};

