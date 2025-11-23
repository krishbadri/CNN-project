#include "cuda_utils.cuh"
#include "activations.cuh"
#include <cuda_runtime.h>

/**
 * CUDA kernel for ReLU activation
 * Element-wise: output[i] = max(0, input[i])
 */
__global__ void relu_forward_kernel(
    const float* input,
    float* output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * Wrapper function to launch ReLU kernel
 */
void relu_forward_gpu(
    const float* d_input,
    float* d_output,
    int size) {
    
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_forward_kernel<<<num_blocks, threads_per_block>>>(
        d_input, d_output, size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

