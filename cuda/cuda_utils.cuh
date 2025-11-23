#pragma once

// Include CUDA runtime - try multiple paths for compatibility
#ifdef __CUDACC__
    #include <cuda_runtime.h>
#else
    // For C++ compilation, try standard path first, then Colab path
    #if __has_include(<cuda_runtime.h>)
        #include <cuda_runtime.h>
    #elif __has_include("/usr/local/cuda/include/cuda_runtime.h")
        #include "/usr/local/cuda/include/cuda_runtime.h"
    #else
        // Fallback: assume CUDA is in standard location
        #include <cuda_runtime.h>
    #endif
#endif

#include <iostream>
#include <cassert>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err_) << std::endl; \
            assert(false); \
        } \
    } while(0)

// Allocate device memory
inline float* device_alloc(size_t n) {
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed (" << n << " floats): " 
                  << cudaGetErrorString(err) << std::endl;
        return nullptr;  // Return nullptr instead of asserting
    }
    return ptr;
}

// Free device memory
inline void device_free(float* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Copy host to device
inline void copy_to_device(float* d, const float* h, size_t n) {
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
}

// Copy device to host
inline void copy_to_host(float* h, const float* d, size_t n) {
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// Create CUDA event for timing
inline cudaEvent_t create_event() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

inline void destroy_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

inline void record_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventRecord(event));
}

inline float elapsed_time_ms(cudaEvent_t start, cudaEvent_t end) {
    // Ensure events are ready before querying
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

