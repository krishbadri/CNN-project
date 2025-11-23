#include "../../include/layers/softmax.hpp"
#include <cmath>
#include <algorithm>

Tensor SoftmaxLayer::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int N = input_shape[0];
    int D = input_shape[1];
    
    Tensor output(input_shape);
    output.compute_strides();
    
    for (int n = 0; n < N; ++n) {
        // Find max for numerical stability
        float max_val = input(n, 0);
        for (int d = 1; d < D; ++d) {
            max_val = std::max(max_val, input(n, d));
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            output(n, d) = std::exp(input(n, d) - max_val);
            sum += output(n, d);
        }
        
        // Normalize
        for (int d = 0; d < D; ++d) {
            output(n, d) /= sum;
        }
    }
    
    return output;
}

