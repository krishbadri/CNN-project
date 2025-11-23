#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>

/**
 * Simple Tensor class for multi-dimensional arrays
 * Stores data in row-major order (NCHW format)
 */
class Tensor {
public:
    Tensor() : shape_{}, data_{} {}
    
    Tensor(const std::vector<int>& shape) : shape_(shape) {
        int total = 1;
        for (int s : shape) {
            if (s <= 0) {
                std::cerr << "Error: Invalid tensor shape dimension: " << s << std::endl;
                data_.clear();
                return;
            }
            total *= s;
        }
        data_.resize(total, 0.0f);
        compute_strides();  // Initialize strides automatically
    }
    
    Tensor(const std::vector<int>& shape, const std::vector<float>& data) 
        : shape_(shape), data_(data) {}
    
    // Access element at indices (with bounds checking in debug mode)
    float& operator()(int n, int c, int h, int w) {
        #ifndef NDEBUG
        if (shape_.size() != 4 || n < 0 || n >= shape_[0] || c < 0 || c >= shape_[1] ||
            h < 0 || h >= shape_[2] || w < 0 || w >= shape_[3]) {
            std::cerr << "Tensor index out of bounds: (" << n << ", " << c << ", " << h << ", " << w << ")" << std::endl;
            std::cerr << "Tensor shape: (" << shape_[0] << ", " << shape_[1] << ", " << shape_[2] << ", " << shape_[3] << ")" << std::endl;
        }
        #endif
        return data_[n * stride_[0] + c * stride_[1] + h * stride_[2] + w * stride_[3]];
    }
    
    const float& operator()(int n, int c, int h, int w) const {
        #ifndef NDEBUG
        if (shape_.size() != 4 || n < 0 || n >= shape_[0] || c < 0 || c >= shape_[1] ||
            h < 0 || h >= shape_[2] || w < 0 || w >= shape_[3]) {
            std::cerr << "Tensor index out of bounds: (" << n << ", " << c << ", " << h << ", " << w << ")" << std::endl;
            std::cerr << "Tensor shape: (" << shape_[0] << ", " << shape_[1] << ", " << shape_[2] << ", " << shape_[3] << ")" << std::endl;
        }
        #endif
        return data_[n * stride_[0] + c * stride_[1] + h * stride_[2] + w * stride_[3]];
    }
    
    // 2D access for FC layers
    float& operator()(int n, int d) {
        #ifndef NDEBUG
        if (shape_.size() != 2 || n < 0 || n >= shape_[0] || d < 0 || d >= shape_[1]) {
            std::cerr << "Tensor index out of bounds: (" << n << ", " << d << ")" << std::endl;
            std::cerr << "Tensor shape: (" << shape_[0] << ", " << shape_[1] << ")" << std::endl;
        }
        #endif
        return data_[n * stride_2d_[0] + d * stride_2d_[1]];
    }
    
    const float& operator()(int n, int d) const {
        #ifndef NDEBUG
        if (shape_.size() != 2 || n < 0 || n >= shape_[0] || d < 0 || d >= shape_[1]) {
            std::cerr << "Tensor index out of bounds: (" << n << ", " << d << ")" << std::endl;
            std::cerr << "Tensor shape: (" << shape_[0] << ", " << shape_[1] << ")" << std::endl;
        }
        #endif
        return data_[n * stride_2d_[0] + d * stride_2d_[1]];
    }
    
    // Raw data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return static_cast<int>(data_.size()); }
    
    // Initialize strides for efficient indexing
    void compute_strides() {
        if (shape_.size() == 4) {
            stride_[3] = 1;
            stride_[2] = shape_[3];
            stride_[1] = shape_[2] * shape_[3];
            stride_[0] = shape_[1] * shape_[2] * shape_[3];
        } else if (shape_.size() == 2) {
            stride_2d_[1] = 1;
            stride_2d_[0] = shape_[1];
        }
    }
    
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    void print_shape() const {
        std::cout << "Tensor shape: (";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }

private:
    std::vector<int> shape_;
    std::vector<float> data_;
    int stride_[4] = {0, 0, 0, 0};
    int stride_2d_[2] = {0, 0};
};

