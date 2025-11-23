#include "cuda_utils.cuh"
#include "pool_kernel.cuh"
#include <cuda_runtime.h>

/**
 * CUDA kernel for max pooling
 * Each thread computes one output pixel
 */
__global__ void maxpool_forward_kernel(
    const float* input,   // (N, C, H_in, W_in)
    float* output,         // (N, C, H_out, W_out)
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pool_size) {
    
    // Flattened thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * H_out * W_out;
    
    if (tid >= total_outputs) {
        return;
    }
    
    // Decompose thread index into (n, c, h_out, w_out)
    int n = tid / (C * H_out * W_out);
    int rem = tid % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;
    
    // Bounds check
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) {
        return;
    }
    
    // Find max in pool window
    float max_val = -1e9f;
    int h_start = h_out * pool_size;
    int w_start = w_out * pool_size;
    
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            
            if (h_in < H_in && w_in < W_in) {
                int input_idx = n * (C * H_in * W_in) +
                               c * (H_in * W_in) +
                               h_in * W_in + w_in;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    // Output index: (n, c, h_out, w_out)
    int output_idx = n * (C * H_out * W_out) +
                     c * (H_out * W_out) +
                     h_out * W_out + w_out;
    
    output[output_idx] = max_val;
}

/**
 * Wrapper function to launch max pooling kernel
 */
void maxpool_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pool_size) {
    
    // Validate parameters
    if (N <= 0 || C <= 0 || H_in <= 0 || W_in <= 0 || 
        H_out <= 0 || W_out <= 0 || pool_size <= 0) {
        std::cerr << "Error: Invalid pooling parameters" << std::endl;
        return;
    }
    
    // Total number of output elements
    int total_outputs = N * C * H_out * W_out;
    
    // Use 1D grid with threads per block
    int threads_per_block = 256;
    int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    // Check for reasonable grid size
    if (num_blocks > 65535) {
        std::cerr << "Error: Grid size too large: " << num_blocks << " blocks" << std::endl;
        return;
    }
    
    maxpool_forward_kernel<<<num_blocks, threads_per_block>>>(
        d_input, d_output,
        N, C, H_in, W_in, H_out, W_out, pool_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

