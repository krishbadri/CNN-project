#pragma once

#include "tensor.hpp"
#include <vector>
#include <string>

/**
 * Load MNIST dataset from IDX format files
 */
class MNISTLoader {
public:
    // Load images from IDX3-ubyte file
    static std::vector<Tensor> load_images(const std::string& filename, int max_images = -1);
    
    // Load labels from IDX1-ubyte file
    static std::vector<int> load_labels(const std::string& filename, int max_labels = -1);
    
    // Helper: read 32-bit integer in big-endian format
    static int read_int32(std::ifstream& file);
};

