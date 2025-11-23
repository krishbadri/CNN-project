#include "../../include/layers/fc_layer.hpp"
#include <cassert>

FCLayer::FCLayer(int in_features, int out_features,
                 const std::vector<float>& weights, const std::vector<float>& bias)
    : in_features_(in_features), out_features_(out_features),
      weights_(weights), bias_(bias) {
    assert(weights.size() == static_cast<size_t>(out_features * in_features));
    assert(bias.size() == static_cast<size_t>(out_features));
}

Tensor FCLayer::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int N = input_shape[0];
    
    // Handle both 2D (N, D_in) and 4D (N, C, H, W) inputs
    int D_in;
    if (input_shape.size() == 2) {
        D_in = input_shape[1];
    } else if (input_shape.size() == 4) {
        D_in = input_shape[1] * input_shape[2] * input_shape[3];
    } else {
        assert(false && "Invalid input shape for FC layer");
    }
    
    assert(D_in == in_features_);
    
    Tensor output({N, out_features_});
    output.compute_strides();
    
    // Matrix-vector multiplication: output = input @ weights^T + bias
    for (int n = 0; n < N; ++n) {
        for (int j = 0; j < out_features_; ++j) {
            float sum = bias_[j];
            
            for (int i = 0; i < in_features_; ++i) {
                float input_val;
                if (input_shape.size() == 2) {
                    input_val = input(n, i);
                } else {
                    // Flatten 4D to 1D index
                    int c = i / (input_shape[2] * input_shape[3]);
                    int rem = i % (input_shape[2] * input_shape[3]);
                    int h = rem / input_shape[3];
                    int w = rem % input_shape[3];
                    input_val = input(n, c, h, w);
                }
                
                int weight_idx = j * in_features_ + i;
                sum += input_val * weights_[weight_idx];
            }
            
            output(n, j) = sum;
        }
    }
    
    return output;
}

