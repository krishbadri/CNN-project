cmake_minimum_required(VERSION 3.18)

# Try to enable CUDA, but make it optional for local development
option(ENABLE_CUDA "Enable CUDA support" ON)

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        message(STATUS "CUDA found - GPU acceleration enabled")
        set(USE_CUDA TRUE)
    else()
        message(WARNING "CUDA not found - building CPU-only version")
        set(USE_CUDA FALSE)
    endif()
else()
    message(STATUS "CUDA disabled - building CPU-only version")
    set(USE_CUDA FALSE)
endif()

project(mnist_cnn_cuda LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

# Add CUDA include paths if CUDA is available
if(USE_CUDA)
    if(CUDA_TOOLKIT_ROOT_DIR)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} REALPATH)
        include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
    endif()
    if(CUDA_INCLUDE_DIRS)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
    # Colab path
    include_directories(/usr/local/cuda/include)
endif()

# Source files
set(CPU_SOURCES
    src/main.cpp
    src/model.cpp
    src/mnist_loader.cpp
    src/layers/conv_layer.cpp
    src/layers/relu_layer.cpp
    src/layers/pool_layer.cpp
    src/layers/fc_layer.cpp
    src/layers/softmax.cpp
)

# Create executable
if(USE_CUDA)
    set(CUDA_SOURCES
        cuda/conv_kernel.cu
        cuda/fc_kernel.cu
        cuda/activations.cu
        cuda/pool_kernel.cu
        cuda/softmax_kernel.cu
    )
    add_executable(mnist_cnn ${CPU_SOURCES} ${CUDA_SOURCES})
    
    # Find CUDA runtime library
    find_library(CUDART_LIBRARY 
        NAMES cudart
        PATHS 
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
        REQUIRED
    )
    
    # Link CUDA libraries
    target_link_libraries(mnist_cnn 
        ${CUDA_LIBRARIES} 
        ${CUDART_LIBRARY}
    )
    
    # Set CUDA architecture (Colab T4 = 75, V100 = 70, A100 = 80)
    set_property(TARGET mnist_cnn PROPERTY CUDA_ARCHITECTURES "70;75;80;86;89")
    
    # Define preprocessor macro
    target_compile_definitions(mnist_cnn PRIVATE USE_CUDA=1)
else()
    add_executable(mnist_cnn ${CPU_SOURCES})
    target_compile_definitions(mnist_cnn PRIVATE USE_CUDA=0)
endif()

# Compiler flags
target_compile_options(mnist_cnn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
