#include "../../include/layers/conv_layer.hpp"
#include <cassert>

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size,
                     const std::vector<float>& weights, const std::vector<float>& bias)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
      weights_(weights), bias_(bias) {
    assert(weights.size() == static_cast<size_t>(out_channels * in_channels * kernel_size * kernel_size));
    assert(bias.size() == static_cast<size_t>(out_channels));
}

Tensor ConvLayer::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int N = input_shape[0];
    int C_in = input_shape[1];
    int H_in = input_shape[2];
    int W_in = input_shape[3];
    
    assert(C_in == in_channels_);
    
    int H_out = H_in - kernel_size_ + 1;
    int W_out = W_in - kernel_size_ + 1;
    
    Tensor output({N, out_channels_, H_out, W_out});
    output.compute_strides();
    
    // Naive convolution: for each output pixel, compute dot product
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels_; ++c_out) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    float sum = bias_[c_out];
                    
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int h_in = h_out + kh;
                                int w_in = w_out + kw;
                                
                                int weight_idx = c_out * (in_channels_ * kernel_size_ * kernel_size_) +
                                                c_in * (kernel_size_ * kernel_size_) +
                                                kh * kernel_size_ + kw;
                                
                                sum += input(n, c_in, h_in, w_in) * weights_[weight_idx];
                            }
                        }
                    }
                    
                    output(n, c_out, h_out, w_out) = sum;
                }
            }
        }
    }
    
    return output;
}

