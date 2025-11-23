#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA kernel for fully-connected layer forward pass
 */
void fc_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int D_in, int D_out);

#else
// Forward declaration for CPU code
void fc_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int D_in, int D_out);
#endif

