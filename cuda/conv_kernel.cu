#include "cuda_utils.cuh"
#include "conv_kernel.cuh"
#include <cuda_runtime.h>

/**
 * CUDA kernel for convolution forward pass
 * Each thread computes one output pixel (n, c_out, h_out, w_out)
 */
__global__ void conv_forward_kernel(
    const float* input,   // (N, C_in, H, W)
    const float* weight,  // (C_out, C_in, K, K)
    const float* bias,    // (C_out)
    float* output,        // (N, C_out, H_out, W_out)
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out) {
    
    // Flattened thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C_out * H_out * W_out;
    
    if (tid >= total_outputs) {
        return;
    }
    
    // Decompose thread index into (n, c_out, h_out, w_out)
    int n = tid / (C_out * H_out * W_out);
    int rem = tid % (C_out * H_out * W_out);
    int c_out = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;
    
    // Bounds check
    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out) {
        return;
    }
    
    // Compute convolution for this output pixel
    float sum = bias[c_out];
    
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                
                // Explicit bounds check (defensive programming)
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    // Input index: (n, c_in, h_in, w_in)
                    int input_idx = n * (C_in * H * W) + 
                                   c_in * (H * W) + 
                                   h_in * W + w_in;
                    
                    // Weight index: (c_out, c_in, kh, kw)
                    int weight_idx = c_out * (C_in * K * K) +
                                    c_in * (K * K) +
                                    kh * K + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Output index: (n, c_out, h_out, w_out)
    int output_idx = n * (C_out * H_out * W_out) +
                     c_out * (H_out * W_out) +
                     h_out * W_out + w_out;
    
    output[output_idx] = sum;
}

/**
 * Wrapper function to launch convolution kernel
 */
void conv_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out) {
    
    // Validate parameters
    if (N <= 0 || C_in <= 0 || H <= 0 || W <= 0 || C_out <= 0 || K <= 0 || 
        H_out <= 0 || W_out <= 0) {
        std::cerr << "Error: Invalid convolution parameters" << std::endl;
        return;
    }
    
    // Total number of output elements
    int total_outputs = N * C_out * H_out * W_out;
    
    // Use 1D grid with threads per block
    int threads_per_block = 256;
    int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    // Check for reasonable grid size (prevent overflow)
    if (num_blocks > 65535) {  // Max grid dimension
        std::cerr << "Error: Grid size too large: " << num_blocks << " blocks" << std::endl;
        return;
    }
    
    conv_forward_kernel<<<num_blocks, threads_per_block>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K, H_out, W_out
    );
    
    CUDA_CHECK(cudaGetLastError());
}

