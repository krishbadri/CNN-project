#include "../include/model.hpp"
#ifdef USE_CUDA
#include "../cuda/conv_kernel.cuh"
#include "../cuda/fc_kernel.cuh"
#include "../cuda/activations.cuh"
#include "../cuda/pool_kernel.cuh"
#include "../cuda/softmax_kernel.cuh"
#include "../cuda/cuda_utils.cuh"
#endif
#include <chrono>
#include <random>
#include <cmath>
#include <fstream>

CNNModel::CNNModel()
    : relu_layer_(),
      pool_layer_(2),      // 2x2 max pool
      softmax_layer_() {
    // Initialize layers with architecture:
    // Conv(1 -> 8 channels, 5x5 kernel) -> ReLU -> MaxPool(2x2) -> FC(1152 -> 10) -> Softmax
    
    // After conv: (N, 8, 24, 24)  [28-5+1 = 24]
    // After pool: (N, 8, 12, 12)  [24/2 = 12]
    // Flatten: (N, 8*12*12) = (N, 1152)
    // FC: (N, 10)
    
    // Initialize with random weights for now (can load from file later)
    init_random_weights();
}

CNNModel::~CNNModel() {
    free_gpu_memory();
}

void CNNModel::init_random_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    // Conv layer weights: (8, 1, 5, 5) = 200
    std::vector<float> conv_weights(8 * 1 * 5 * 5);
    std::vector<float> conv_bias(8);
    for (float& w : conv_weights) w = dist(gen);
    for (float& b : conv_bias) b = dist(gen);
    
    conv_layer_ = std::make_unique<ConvLayer>(1, 8, 5, conv_weights, conv_bias);
    
    // FC layer weights: (10, 1152) = 11520
    std::vector<float> fc_weights(10 * 1152);
    std::vector<float> fc_bias(10);
    for (float& w : fc_weights) w = dist(gen);
    for (float& b : fc_bias) b = dist(gen);
    
    fc_layer_ = std::make_unique<FCLayer>(1152, 10, fc_weights, fc_bias);
}

bool CNNModel::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weights file " << filename << std::endl;
        return false;
    }
    
    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t expected_size = (200 + 8 + 11520 + 10) * sizeof(float);
    if (file_size < expected_size) {
        std::cerr << "Error: Weights file too small. Expected " << expected_size 
                  << " bytes, got " << file_size << " bytes" << std::endl;
        file.close();
        return false;
    }
    
    // Conv weights: (8, 1, 5, 5) = 200 floats
    std::vector<float> conv_weights(200);
    if (!file.read(reinterpret_cast<char*>(conv_weights.data()), 200 * sizeof(float))) {
        std::cerr << "Error: Failed to read conv weights" << std::endl;
        file.close();
        return false;
    }
    
    // Conv bias: (8,) = 8 floats
    std::vector<float> conv_bias(8);
    if (!file.read(reinterpret_cast<char*>(conv_bias.data()), 8 * sizeof(float))) {
        std::cerr << "Error: Failed to read conv bias" << std::endl;
        file.close();
        return false;
    }
    
    // FC weights: (10, 1152) = 11520 floats
    std::vector<float> fc_weights(11520);
    if (!file.read(reinterpret_cast<char*>(fc_weights.data()), 11520 * sizeof(float))) {
        std::cerr << "Error: Failed to read FC weights" << std::endl;
        file.close();
        return false;
    }
    
    // FC bias: (10,) = 10 floats
    std::vector<float> fc_bias(10);
    if (!file.read(reinterpret_cast<char*>(fc_bias.data()), 10 * sizeof(float))) {
        std::cerr << "Error: Failed to read FC bias" << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    
    // Recreate layers with loaded weights
    conv_layer_ = std::make_unique<ConvLayer>(1, 8, 5, conv_weights, conv_bias);
    fc_layer_ = std::make_unique<FCLayer>(1152, 10, fc_weights, fc_bias);
    
    std::cout << "Loaded weights from " << filename << std::endl;
    return true;
}

Tensor CNNModel::forward_cpu(const Tensor& input) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Conv
    auto start = std::chrono::high_resolution_clock::now();
    Tensor conv_out = conv_layer_->forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    cpu_timings_.conv_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // ReLU
    start = std::chrono::high_resolution_clock::now();
    Tensor relu_out = relu_layer_.forward(conv_out);
    end = std::chrono::high_resolution_clock::now();
    cpu_timings_.relu_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Pool
    start = std::chrono::high_resolution_clock::now();
    Tensor pool_out = pool_layer_.forward(relu_out);
    end = std::chrono::high_resolution_clock::now();
    cpu_timings_.pool_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Flatten and FC
    start = std::chrono::high_resolution_clock::now();
    Tensor fc_out = fc_layer_->forward(pool_out);
    end = std::chrono::high_resolution_clock::now();
    cpu_timings_.fc_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Softmax
    start = std::chrono::high_resolution_clock::now();
    Tensor output = softmax_layer_.forward(fc_out);
    end = std::chrono::high_resolution_clock::now();
    cpu_timings_.softmax_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    auto end_total = std::chrono::high_resolution_clock::now();
    cpu_timings_.total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    return output;
}

#ifdef USE_CUDA
Tensor CNNModel::forward_gpu(const Tensor& input) {
    const auto& input_shape = input.shape();
    
    // Validate input shape
    if (input_shape.size() != 4 || input_shape[0] <= 0 || input_shape[1] != 1 || 
        input_shape[2] != 28 || input_shape[3] != 28) {
        std::cerr << "Error: Invalid input shape for GPU forward pass. Expected (N, 1, 28, 28), got (";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cerr << input_shape[i];
            if (i < input_shape.size() - 1) std::cerr << ", ";
        }
        std::cerr << ")" << std::endl;
        return Tensor({input_shape[0], 10});  // Return dummy output
    }
    
    int N = input_shape[0];
    int C_in = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];
    
    // Allocate GPU memory if needed
    allocate_gpu_memory(N);
    copy_weights_to_gpu();
    
    // Create CUDA events for timing
    cudaEvent_t start_total, end_total;
    cudaEvent_t start, end;
    start_total = create_event();
    end_total = create_event();
    start = create_event();
    end = create_event();
    
    record_event(start_total);
    
    // Copy input to GPU
    copy_to_device(d_conv_input_, input.data(), N * C_in * H * W);
    
    // Conv layer
    record_event(start);
    int C_out = 8;
    int K = 5;
    int H_out = H - K + 1;  // 24
    int W_out = W - K + 1;  // 24
    conv_forward_gpu(d_conv_input_, d_conv_weight_, d_conv_bias_, d_conv_output_,
                     N, C_in, H, W, C_out, K, H_out, W_out);
    cudaDeviceSynchronize();
    record_event(end);
    gpu_timings_.conv_ms = elapsed_time_ms(start, end);
    
    // ReLU (in-place on conv output)
    record_event(start);
    relu_forward_gpu(d_conv_output_, d_relu_output_, N * C_out * H_out * W_out);
    cudaDeviceSynchronize();
    record_event(end);
    gpu_timings_.relu_ms = elapsed_time_ms(start, end);
    
    // Pool layer (GPU)
    record_event(start);
    int H_pool = H_out / 2;  // 12
    int W_pool = W_out / 2;  // 12
    maxpool_forward_gpu(d_relu_output_, d_pool_output_,
                       N, C_out, H_out, W_out, H_pool, W_pool, 2);
    cudaDeviceSynchronize();
    record_event(end);
    gpu_timings_.pool_ms = elapsed_time_ms(start, end);
    
    // Flatten and FC
    record_event(start);
    int D_in = C_out * H_pool * W_pool;  // 1152
    int D_out = 10;
    fc_forward_gpu(d_pool_output_, d_fc_weight_, d_fc_bias_, d_fc_output_,
                   N, D_in, D_out);
    cudaDeviceSynchronize();
    record_event(end);
    gpu_timings_.fc_ms = elapsed_time_ms(start, end);
    
    // Softmax (GPU)
    record_event(start);
    softmax_forward_gpu(d_fc_output_, d_softmax_output_, N, D_out);
    cudaDeviceSynchronize();
    record_event(end);
    gpu_timings_.softmax_ms = elapsed_time_ms(start, end);
    
    record_event(end_total);
    cudaDeviceSynchronize();  // Ensure all GPU work is done before timing
    gpu_timings_.total_ms = elapsed_time_ms(start_total, end_total);
    
    // Copy final output back to CPU
    Tensor output({N, D_out});
    copy_to_host(output.data(), d_softmax_output_, N * D_out);
    output.compute_strides();
    
    destroy_event(start_total);
    destroy_event(end_total);
    destroy_event(start);
    destroy_event(end);
    
    return output;
}
#else
Tensor CNNModel::forward_gpu(const Tensor& input) {
    std::cerr << "Error: GPU forward pass called but CUDA not available!" << std::endl;
    return Tensor({1, 10});  // Return dummy tensor
}
#endif

#ifdef USE_CUDA
void CNNModel::allocate_gpu_memory(int batch_size) {
    // Validate batch size
    if (batch_size <= 0) {
        std::cerr << "Error: Invalid batch size: " << batch_size << std::endl;
        return;
    }
    
    // Conv layer buffers
    int conv_input_size = batch_size * 1 * 28 * 28;
    int conv_output_size = batch_size * 8 * 24 * 24;
    int conv_weight_size = 8 * 1 * 5 * 5;
    int conv_bias_size = 8;
    
    if (!d_conv_input_) {
        d_conv_input_ = device_alloc(conv_input_size);
        d_conv_output_ = device_alloc(conv_output_size);
        d_conv_weight_ = device_alloc(conv_weight_size);
        d_conv_bias_ = device_alloc(conv_bias_size);
        
        d_relu_output_ = device_alloc(conv_output_size);
        d_pool_output_ = device_alloc(batch_size * 8 * 12 * 12);
        
        // Check for allocation failures
        if (!d_conv_input_ || !d_conv_output_ || !d_conv_weight_ || !d_conv_bias_ ||
            !d_relu_output_ || !d_pool_output_) {
            std::cerr << "Error: Failed to allocate GPU memory for conv layer" << std::endl;
            free_gpu_memory();  // Clean up any partial allocations
            return;
        }
    }
    
    // FC layer buffers
    int fc_input_size = batch_size * 1152;
    int fc_output_size = batch_size * 10;
    int fc_weight_size = 10 * 1152;
    int fc_bias_size = 10;
    
    if (!d_fc_input_) {
        d_fc_input_ = device_alloc(fc_input_size);
        d_fc_output_ = device_alloc(fc_output_size);
        d_fc_weight_ = device_alloc(fc_weight_size);
        d_fc_bias_ = device_alloc(fc_bias_size);
        d_softmax_output_ = device_alloc(fc_output_size);  // Same size as FC output
        
        // Check for allocation failures
        if (!d_fc_input_ || !d_fc_output_ || !d_fc_weight_ || !d_fc_bias_ || !d_softmax_output_) {
            std::cerr << "Error: Failed to allocate GPU memory for FC layer" << std::endl;
            free_gpu_memory();  // Clean up any partial allocations
            return;
        }
    }
}

void CNNModel::copy_weights_to_gpu() {
    if (!d_conv_weight_) return;
    
    const auto& conv_weights = conv_layer_->get_weights();
    const auto& conv_bias = conv_layer_->get_bias();
    copy_to_device(d_conv_weight_, conv_weights.data(), conv_weights.size());
    copy_to_device(d_conv_bias_, conv_bias.data(), conv_bias.size());
    
    const auto& fc_weights = fc_layer_->get_weights();
    const auto& fc_bias = fc_layer_->get_bias();
    copy_to_device(d_fc_weight_, fc_weights.data(), fc_weights.size());
    copy_to_device(d_fc_bias_, fc_bias.data(), fc_bias.size());
}
#else
void CNNModel::allocate_gpu_memory(int batch_size) {
    // No-op when CUDA not available
}

void CNNModel::copy_weights_to_gpu() {
    // No-op when CUDA not available
}
#endif

void CNNModel::free_gpu_memory() {
#ifdef USE_CUDA
    device_free(d_conv_input_);
    device_free(d_conv_output_);
    device_free(d_conv_weight_);
    device_free(d_conv_bias_);
    device_free(d_relu_output_);
    device_free(d_pool_output_);
    device_free(d_fc_input_);
    device_free(d_fc_output_);
    device_free(d_fc_weight_);
    device_free(d_fc_bias_);
    device_free(d_softmax_output_);
#endif
    // Reset pointers
    d_conv_input_ = nullptr;
    d_conv_output_ = nullptr;
    d_conv_weight_ = nullptr;
    d_conv_bias_ = nullptr;
    d_relu_output_ = nullptr;
    d_pool_output_ = nullptr;
    d_fc_input_ = nullptr;
    d_fc_output_ = nullptr;
    d_fc_weight_ = nullptr;
    d_fc_bias_ = nullptr;
    d_softmax_output_ = nullptr;
}

