#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA kernel for ReLU activation
 */
void relu_forward_gpu(
    const float* d_input,
    float* d_output,
    int size);

#else
// Forward declaration for CPU code
void relu_forward_gpu(
    const float* d_input,
    float* d_output,
    int size);
#endif

