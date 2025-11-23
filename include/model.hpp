#pragma once

#include "tensor.hpp"
#include "layers/conv_layer.hpp"
#include "layers/relu_layer.hpp"
#include "layers/pool_layer.hpp"
#include "layers/fc_layer.hpp"
#include "layers/softmax.hpp"
#include <vector>
#include <memory>
#include <string>

/**
 * CNN Model that supports both CPU and GPU inference
 * Architecture: Conv(8, 5x5) -> ReLU -> MaxPool(2x2) -> FC(1152 -> 10) -> Softmax
 */
class CNNModel {
public:
    CNNModel();
    ~CNNModel();
    
    // CPU forward pass
    Tensor forward_cpu(const Tensor& input);
    
    // GPU forward pass
    Tensor forward_gpu(const Tensor& input);
    
    // Get layer timings (for benchmarking)
    struct LayerTimings {
        double conv_ms = 0.0;
        double relu_ms = 0.0;
        double pool_ms = 0.0;
        double fc_ms = 0.0;
        double softmax_ms = 0.0;
        double total_ms = 0.0;
    };
    
    LayerTimings get_cpu_timings() const { return cpu_timings_; }
    LayerTimings get_gpu_timings() const { return gpu_timings_; }
    
    // Initialize with random weights (for testing)
    void init_random_weights();
    
    // Load weights from file (trained by Python script)
    bool load_weights(const std::string& filename);

private:
    // CPU layers
    std::unique_ptr<ConvLayer> conv_layer_;
    ReLULayer relu_layer_;
    MaxPoolLayer pool_layer_;
    std::unique_ptr<FCLayer> fc_layer_;
    SoftmaxLayer softmax_layer_;
    
    // GPU device memory pointers
    float* d_conv_input_ = nullptr;
    float* d_conv_output_ = nullptr;
    float* d_conv_weight_ = nullptr;
    float* d_conv_bias_ = nullptr;
    
    float* d_relu_output_ = nullptr;
    float* d_pool_output_ = nullptr;
    
    float* d_fc_input_ = nullptr;
    float* d_fc_output_ = nullptr;
    float* d_fc_weight_ = nullptr;
    float* d_fc_bias_ = nullptr;
    
    float* d_softmax_output_ = nullptr;
    
    // Timing data
    mutable LayerTimings cpu_timings_;
    mutable LayerTimings gpu_timings_;
    
    // Helper: allocate GPU memory
    void allocate_gpu_memory(int batch_size);
    void free_gpu_memory();
    
    // Helper: copy weights to GPU
    void copy_weights_to_gpu();
};

