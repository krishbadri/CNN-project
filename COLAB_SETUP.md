# Google Colab Setup Guide

Complete guide to run this CNN project on Google Colab with GPU acceleration.

## Step 1: Upload Project Files

### Option A: Upload via Colab UI
1. Open Google Colab: https://colab.research.google.com/
2. File → Upload notebook → Upload the project files
3. Or use the file browser on the left sidebar

### Option B: Clone from GitHub (Recommended)
```python
# In a Colab cell
!git clone https://github.com/yourusername/CNN-project.git
%cd CNN-project
```

## Step 2: Install Dependencies

```python
# Install CUDA toolkit (Colab has it, but verify)
!nvcc --version

# Install CMake if needed
!apt-get update
!apt-get install -y cmake build-essential

# Verify CUDA is available
!nvidia-smi
```

## Step 3: Download MNIST Dataset

```python
# Download MNIST files
!mkdir -p data
!wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte -O data/train-images-idx3-ubyte
!wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte -O data/train-labels-idx1-ubyte
!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte -O data/t10k-images-idx3-ubyte
!wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte -O data/t10k-labels-idx1-ubyte

# Verify files
!ls -lh data/
```

## Step 4: Train the Model (Optional)

```python
# Install PyTorch
!pip install torch torchvision

# Train the model
!python scripts/train_model.py
```

This will create `weights/model_weights.bin` with trained weights (~98% accuracy).

## Step 5: Build the C++ Project

```python
# Create build directory
!mkdir -p build
%cd build

# Configure with CMake
!cmake ..

# Build (this may take a few minutes)
!make -j4

# Or use cmake build
!cmake --build . --config Release -j4
```

## Step 6: Run Inference

```python
# Run with GPU acceleration
!./mnist_cnn --mode gpu --num 1000

# Or compare CPU vs GPU
!./mnist_cnn --mode both --num 1000
```

## Complete Colab Notebook Template

```python
# Cell 1: Setup
!git clone https://github.com/yourusername/CNN-project.git
%cd CNN-project

# Cell 2: Install dependencies
!apt-get update
!apt-get install -y cmake build-essential
!pip install torch torchvision

# Cell 3: Download MNIST
!mkdir -p data
!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte -O data/t10k-images-idx3-ubyte
!wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte -O data/t10k-labels-idx1-ubyte

# Cell 4: Train model (optional)
!python scripts/train_model.py

# Cell 5: Build C++ project
!mkdir -p build && cd build
!cmake ..
!make -j4

# Cell 6: Run inference
!./mnist_cnn --mode gpu --num 1000
```

## Troubleshooting

### CUDA Not Found
```python
# Check CUDA installation
!nvcc --version
!nvidia-smi

# If CUDA not found, Colab should have it by default
# Try: Runtime → Change runtime type → GPU
```

### CMake Can't Find CUDA
```python
# Set CUDA path explicitly
import os
os.environ['CUDA_PATH'] = '/usr/local/cuda'

# Or in CMake
!cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### Build Errors
```python
# Clean and rebuild
!rm -rf build
!mkdir build && cd build
!cmake ..
!make VERBOSE=1  # See detailed output
```

### MNIST Files Not Found
```python
# Verify files exist
!ls -lh data/

# Re-download if needed
!wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte -O data/t10k-images-idx3-ubyte
```

## Expected Output

```
MNIST CNN Inference - CPU vs GPU Comparison
============================================

Loading MNIST test data...
Loaded 1000 images
Loading trained weights...
Loaded weights from weights/model_weights.bin

Image 0: True label = 7, GPU prediction = 7

=== GPU Timings ===
Conv:      0.234 ms
ReLU:      0.008 ms
Pool:      0.012 ms
FC:        0.089 ms
Softmax:   0.003 ms
Total:     0.346 ms

GPU Accuracy: 981 / 1000 = 98.10%
```

## Notes

- **GPU Runtime**: Make sure Colab runtime is set to GPU (Runtime → Change runtime type)
- **Build Time**: First build takes 2-5 minutes
- **Memory**: Colab GPU has limited memory, batch size is limited
- **All Layers on GPU**: Now all layers (Conv, ReLU, Pool, FC, Softmax) run on GPU!

## Quick Test

```python
# Minimal test
!cd build && ./mnist_cnn --mode gpu --num 10
```

This should complete in seconds and show GPU timings for all layers.

