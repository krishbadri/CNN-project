#include "../../include/layers/pool_layer.hpp"
#include <algorithm>
#include <cmath>

Tensor MaxPoolLayer::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int N = input_shape[0];
    int C = input_shape[1];
    int H_in = input_shape[2];
    int W_in = input_shape[3];
    
    int H_out = H_in / pool_size_;
    int W_out = W_in / pool_size_;
    
    Tensor output({N, C, H_out, W_out});
    output.compute_strides();
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    float max_val = -1e9f;
                    
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            int h_in = h_out * pool_size_ + ph;
                            int w_in = w_out * pool_size_ + pw;
                            max_val = std::max(max_val, input(n, c, h_in, w_in));
                        }
                    }
                    
                    output(n, c, h_out, w_out) = max_val;
                }
            }
        }
    }
    
    return output;
}

