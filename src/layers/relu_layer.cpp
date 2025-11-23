#include "../../include/layers/relu_layer.hpp"

Tensor ReLULayer::forward(const Tensor& input) {
    Tensor output(input.shape());
    output.compute_strides();
    
    for (int i = 0; i < input.size(); ++i) {
        output.data()[i] = std::max(0.0f, input.data()[i]);
    }
    
    return output;
}

