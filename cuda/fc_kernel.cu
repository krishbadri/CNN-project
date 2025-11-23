#include "cuda_utils.cuh"
#include "fc_kernel.cuh"
#include <cuda_runtime.h>

/**
 * CUDA kernel for fully-connected layer forward pass
 * Each thread computes one output neuron (n, j)
 */
__global__ void fc_forward_kernel(
    const float* input,   // (N, D_in) - flattened
    const float* weight,  // (D_out, D_in) - row-major
    const float* bias,    // (D_out)
    float* output,        // (N, D_out)
    int N, int D_in, int D_out) {
    
    // Get thread indices
    int n = blockIdx.x;
    int j = threadIdx.x;
    
    // Bounds check
    if (n >= N || j >= D_out) {
        return;
    }
    
    // Compute dot product: output[n, j] = input[n] @ weight[j] + bias[j]
    float sum = bias[j];
    
    for (int i = 0; i < D_in; ++i) {
        int input_idx = n * D_in + i;
        int weight_idx = j * D_in + i;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    int output_idx = n * D_out + j;
    output[output_idx] = sum;
}

/**
 * Wrapper function to launch FC kernel
 */
void fc_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int D_in, int D_out) {
    
    // Validate parameters
    if (D_out > 1024) {
        std::cerr << "Error: D_out (" << D_out << ") exceeds max threads per block (1024)" << std::endl;
        return;
    }
    
    // Grid: (N, 1, 1)
    // Block: (D_out, 1, 1)
    dim3 block_size(D_out, 1, 1);
    dim3 grid_size(N, 1, 1);
    
    fc_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_weight, d_bias, d_output,
        N, D_in, D_out
    );
    
    CUDA_CHECK(cudaGetLastError());
}

