#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA kernel for softmax forward pass
 */
void softmax_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int D);

#else
// Forward declaration for CPU code
void softmax_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int D);
#endif

