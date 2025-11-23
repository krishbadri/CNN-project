#include "../include/mnist_loader.hpp"
#include <fstream>
#include <iostream>

int MNISTLoader::read_int32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    
    if (file.gcount() != 4) {
        std::cerr << "Error: Failed to read 4 bytes from file (only read " << file.gcount() << " bytes)" << std::endl;
        return 0;
    }
    
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<Tensor> MNISTLoader::load_images(const std::string& filename, int max_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }
    
    // Read magic number
    int magic = read_int32(file);
    if (magic != 2051) {
        std::cerr << "Error: Invalid magic number for images file" << std::endl;
        return {};
    }
    
    int num_images = read_int32(file);
    int num_rows = read_int32(file);
    int num_cols = read_int32(file);
    
    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }
    
    std::vector<Tensor> images;
    images.reserve(num_images);
    
    for (int i = 0; i < num_images; ++i) {
        Tensor img({1, 1, num_rows, num_cols});
        unsigned char pixel;
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                if (!file.read(reinterpret_cast<char*>(&pixel), 1)) {
                    std::cerr << "Error: Unexpected end of file while reading image " << i << std::endl;
                    file.close();
                    return images;  // Return what we've loaded so far
                }
                img(0, 0, r, c) = static_cast<float>(pixel) / 255.0f;
            }
        }
        img.compute_strides();
        images.push_back(img);
    }
    
    file.close();
    return images;
}

std::vector<int> MNISTLoader::load_labels(const std::string& filename, int max_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }
    
    // Read magic number
    int magic = read_int32(file);
    if (magic != 2049) {
        std::cerr << "Error: Invalid magic number for labels file" << std::endl;
        return {};
    }
    
    int num_labels = read_int32(file);
    
    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }
    
    std::vector<int> labels;
    labels.reserve(num_labels);
    
    unsigned char label;
    for (int i = 0; i < num_labels; ++i) {
        if (!file.read(reinterpret_cast<char*>(&label), 1)) {
            std::cerr << "Error: Unexpected end of file while reading label " << i << std::endl;
            file.close();
            return labels;  // Return what we've loaded so far
        }
        labels.push_back(static_cast<int>(label));
    }
    
    file.close();
    return labels;
}

