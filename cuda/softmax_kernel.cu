#include "cuda_utils.cuh"
#include "softmax_kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>

/**
 * CUDA kernel for softmax
 * Each thread handles one sample, computes softmax over all classes
 */
__global__ void softmax_forward_kernel(
    const float* input,   // (N, D) - logits
    float* output,        // (N, D) - probabilities
    int N, int D) {
    
    int n = blockIdx.x;
    int d = threadIdx.x;
    
    // Bounds check
    if (n >= N || d >= D) {
        return;
    }
    
    // Shared memory for max and sum (one per sample)
    __shared__ float s_max[256];  // Max per sample (assuming N <= 256)
    __shared__ float s_sum[256];  // Sum per sample
    
    int tid = threadIdx.x;
    
    // Find max for this sample (cooperative reduction)
    float val = input[n * D + d];
    
    // Reduce to find max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && d + stride < D) {
            val = fmaxf(val, input[n * D + d + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_max[n % 256] = val;  // Store max for this sample
    }
    __syncthreads();
    
    float max_val = s_max[n % 256];
    
    // Compute exp(x - max) and sum
    float exp_val = expf(input[n * D + d] - max_val);
    output[n * D + d] = exp_val;
    
    // Reduce to find sum
    val = exp_val;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && d + stride < D) {
            val += output[n * D + d + stride];
        }
        __syncthreads();
        if (tid < stride && d + stride < D) {
            output[n * D + d + stride] = 0;  // Clear for next iteration
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_sum[n % 256] = val;
    }
    __syncthreads();
    
    float sum_val = s_sum[n % 256];
    
    // Normalize
    output[n * D + d] = exp_val / sum_val;
}

/**
 * Simplified softmax kernel - each thread handles one element
 * For small D (10), we use proper reduction with shared memory
 */
__global__ void softmax_forward_kernel_simple(
    const float* input,   // (N, D) - logits
    float* output,        // (N, D) - probabilities
    int N, int D) {
    
    int n = blockIdx.x;
    int d = threadIdx.x;
    
    // Bounds check
    if (n >= N || d >= D) {
        return;
    }
    
    // Shared memory for reduction (one value per thread)
    __shared__ float s_data[256];  // Max 256 threads per block
    __shared__ float s_max[1];     // Store max value
    __shared__ float s_sum[1];     // Store sum value
    
    // Step 1: Load input into shared memory and find max
    s_data[d] = (d < D) ? input[n * D + d] : -1e9f;
    __syncthreads();
    
    // Reduction to find max (proper parallel reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride && d + stride < D) {
            s_data[d] = fmaxf(s_data[d], s_data[d + stride]);
        }
        __syncthreads();
    }
    
    if (d == 0) {
        s_max[0] = s_data[0];
    }
    __syncthreads();
    float max_val = s_max[0];
    
    // Step 2: Compute exp(x - max) and store in shared memory
    float exp_val = (d < D) ? expf(input[n * D + d] - max_val) : 0.0f;
    s_data[d] = exp_val;
    __syncthreads();
    
    // Step 3: Reduction to find sum (proper parallel reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride && d + stride < D) {
            s_data[d] += s_data[d + stride];
        }
        __syncthreads();
    }
    
    if (d == 0) {
        s_sum[0] = s_data[0];
    }
    __syncthreads();
    float sum_val = s_sum[0];
    
    // Step 4: Normalize (avoid division by zero)
    if (d < D) {
        output[n * D + d] = (sum_val > 1e-8f) ? exp_val / sum_val : (1.0f / D);
    }
}

/**
 * Wrapper function to launch softmax kernel
 */
void softmax_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int D) {
    
    // Validate parameters
    if (N <= 0 || D <= 0) {
        std::cerr << "Error: Invalid softmax parameters" << std::endl;
        return;
    }
    
    // Use simple kernel: Grid (N, 1, 1), Block (D, 1, 1)
    // For small D (10), this is fine
    dim3 block_size(D, 1, 1);
    dim3 grid_size(N, 1, 1);
    
    softmax_forward_kernel_simple<<<grid_size, block_size>>>(
        d_input, d_output, N, D
    );
    
    CUDA_CHECK(cudaGetLastError());
}

