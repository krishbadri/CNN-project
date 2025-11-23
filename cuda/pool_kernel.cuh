#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA kernel for max pooling forward pass
 */
void maxpool_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pool_size);

#else
// Forward declaration for CPU code
void maxpool_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pool_size);
#endif

