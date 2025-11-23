#pragma once

#include "../tensor.hpp"

/**
 * Fully connected layer (CPU implementation)
 * Input: (N, D_in) - flattened
 * Output: (N, D_out)
 */
class FCLayer {
public:
    FCLayer(int in_features, int out_features,
            const std::vector<float>& weights, const std::vector<float>& bias);
    
    Tensor forward(const Tensor& input);
    
    int get_in_features() const { return in_features_; }
    int get_out_features() const { return out_features_; }
    
    const std::vector<float>& get_weights() const { return weights_; }
    const std::vector<float>& get_bias() const { return bias_; }

private:
    int in_features_;
    int out_features_;
    std::vector<float> weights_;  // (out_features, in_features) - row-major
    std::vector<float> bias_;     // (out_features)
};

