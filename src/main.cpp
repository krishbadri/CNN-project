#include "../include/model.hpp"
#include "../include/mnist_loader.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

void print_timings(const CNNModel::LayerTimings& timings, const std::string& mode) {
    std::cout << "\n=== " << mode << " Timings ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Conv:      " << timings.conv_ms << " ms" << std::endl;
    std::cout << "ReLU:      " << timings.relu_ms << " ms" << std::endl;
    std::cout << "Pool:      " << timings.pool_ms << " ms" << std::endl;
    std::cout << "FC:        " << timings.fc_ms << " ms" << std::endl;
    std::cout << "Softmax:   " << timings.softmax_ms << " ms" << std::endl;
    std::cout << "Total:     " << timings.total_ms << " ms" << std::endl;
}

bool compare_outputs(const Tensor& cpu_out, const Tensor& gpu_out, float tolerance = 1e-4f) {
    if (cpu_out.shape() != gpu_out.shape()) {
        std::cerr << "Shape mismatch!" << std::endl;
        return false;
    }
    
    int total = cpu_out.size();
    int mismatches = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < total; ++i) {
        float diff = std::abs(cpu_out.data()[i] - gpu_out.data()[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > tolerance) {
            mismatches++;
        }
    }
    
    std::cout << "\n=== Correctness Check ===" << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Mismatches (>" << tolerance << "): " << mismatches << " / " << total << std::endl;
    
    if (max_diff < tolerance) {
        std::cout << "✓ Outputs match!" << std::endl;
        return true;
    } else {
        std::cout << "✗ Outputs differ!" << std::endl;
        return false;
    }
}

int get_prediction(const Tensor& output, int sample_idx = 0) {
    int N = output.shape()[0];
    int D = output.shape()[1];
    
    if (sample_idx < 0 || sample_idx >= N) {
        std::cerr << "Error: Invalid sample index " << sample_idx << " (valid range: 0-" << (N-1) << ")" << std::endl;
        return -1;
    }
    
    int best_class = 0;
    float best_prob = output(sample_idx, 0);
    
    for (int d = 1; d < D; ++d) {
        if (output(sample_idx, d) > best_prob) {
            best_prob = output(sample_idx, d);
            best_class = d;
        }
    }
    
    return best_class;
}

int main(int argc, char* argv[]) {
    std::cout << "MNIST CNN Inference - CPU vs GPU Comparison" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Parse arguments
    std::string mode = "both";
    int batch_size = 1;
    int num_images = 10;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--num" && i + 1 < argc) {
            num_images = std::stoi(argv[++i]);
        }
    }
    
    // Load MNIST test data
    std::cout << "\nLoading MNIST test data..." << std::endl;
    std::vector<Tensor> test_images = MNISTLoader::load_images("data/t10k-images-idx3-ubyte", num_images);
    std::vector<int> test_labels = MNISTLoader::load_labels("data/t10k-labels-idx1-ubyte", num_images);
    
    if (test_images.empty() || test_labels.empty()) {
        std::cerr << "Error: Could not load MNIST data. Make sure data files are in data/ directory." << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << test_images.size() << " images" << std::endl;
    
    // Create model
    CNNModel model;
    
    // Try to load trained weights, otherwise use random
    std::string weights_file = "weights/model_weights.bin";
    if (std::ifstream(weights_file).good()) {
        std::cout << "Loading trained weights..." << std::endl;
        model.load_weights(weights_file);
    } else {
        std::cout << "No trained weights found. Using random weights (accuracy will be ~10%)." << std::endl;
        std::cout << "Run: python scripts/train_model.py to train the model first." << std::endl;
    }
    
    // Process images
    int correct_cpu = 0, correct_gpu = 0;
    
    for (size_t i = 0; i < test_images.size(); ++i) {
        Tensor input = test_images[i];
        input.compute_strides();
        
        int true_label = test_labels[i];
        
        if (mode == "cpu" || mode == "both") {
            Tensor cpu_output = model.forward_cpu(input);
            int pred_cpu = get_prediction(cpu_output);
            if (pred_cpu == true_label) correct_cpu++;
            
            if (i == 0) {
                std::cout << "\nImage " << i << ": True label = " << true_label 
                          << ", CPU prediction = " << pred_cpu << std::endl;
                print_timings(model.get_cpu_timings(), "CPU");
            }
        }
        
#ifdef USE_CUDA
        if (mode == "gpu" || mode == "both") {
            Tensor gpu_output = model.forward_gpu(input);
            int pred_gpu = get_prediction(gpu_output);
            if (pred_gpu == true_label) correct_gpu++;
            
            if (i == 0) {
                std::cout << "\nImage " << i << ": True label = " << true_label 
                          << ", GPU prediction = " << pred_gpu << std::endl;
                print_timings(model.get_gpu_timings(), "GPU");
            }
        }
        
        // Compare outputs if both modes
        if (mode == "both" && i == 0) {
            Tensor cpu_output = model.forward_cpu(input);
            Tensor gpu_output = model.forward_gpu(input);
            compare_outputs(cpu_output, gpu_output);
            
            // Print speedup
            std::cout << "\n=== Speedup ===" << std::endl;
            double speedup = model.get_cpu_timings().total_ms / model.get_gpu_timings().total_ms;
            std::cout << "Total speedup: " << std::fixed << std::setprecision(2) 
                      << speedup << "x" << std::endl;
        }
#else
        if (mode == "gpu" || mode == "both") {
            std::cerr << "Warning: GPU mode requested but CUDA not available. Use --mode cpu" << std::endl;
        }
#endif
    }
    
    // Print accuracy
    if (mode == "cpu" || mode == "both") {
        std::cout << "\nCPU Accuracy: " << correct_cpu << " / " << test_images.size() 
                  << " = " << (100.0 * correct_cpu / test_images.size()) << "%" << std::endl;
    }
    if (mode == "gpu" || mode == "both") {
        std::cout << "GPU Accuracy: " << correct_gpu << " / " << test_images.size() 
                  << " = " << (100.0 * correct_gpu / test_images.size()) << "%" << std::endl;
    }
    
    return 0;
}

