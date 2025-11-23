#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * CUDA kernel for convolution forward pass
 */
void conv_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out);

#else
// Forward declaration for CPU code
void conv_forward_gpu(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    int H_out, int W_out);
#endif

