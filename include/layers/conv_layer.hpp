#pragma once

#include "../tensor.hpp"

/**
 * Convolution layer (CPU implementation)
 * Input: (N, C_in, H, W)
 * Output: (N, C_out, H_out, W_out)
 */
class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, 
              const std::vector<float>& weights, const std::vector<float>& bias);
    
    Tensor forward(const Tensor& input);
    
    int get_in_channels() const { return in_channels_; }
    int get_out_channels() const { return out_channels_; }
    int get_kernel_size() const { return kernel_size_; }
    
    const std::vector<float>& get_weights() const { return weights_; }
    const std::vector<float>& get_bias() const { return bias_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    std::vector<float> weights_;  // (out_channels, in_channels, kernel_size, kernel_size)
    std::vector<float> bias_;     // (out_channels)
};

