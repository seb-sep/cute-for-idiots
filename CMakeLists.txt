cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cutlass_for_idiots LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Set default CUDA architecture if not specified
# 6.1 is pascal for gtx 1070
#6.0 for p100
#8.0 for a100
#8.6 for a40 and other non-a100 ampere
if(NOT DEFINED CUTLASS_NVCC_ARCHS)
    set(CUTLASS_NVCC_ARCHS 86 CACHE STRING "CUDA architectures to compile for")
endif()

# Disable all CUTLASS components except core functionality
option(CUTLASS_ENABLE_TESTS "Enable CUTLASS tests" OFF)
option(CUTLASS_UNITY_BUILD_ENABLED "Enable CUTLASS unity build" ON)
option(CUTLASS_ENABLE_CUBLAS "Enable cuBLAS support" OFF)
option(CUTLASS_ENABLE_CUDNN "Enable cuDNN support" OFF)
option(CUTLASS_ENABLE_LIBRARY "Enable CUTLASS library" OFF)
option(CUTLASS_ENABLE_PROFILER "Enable CUTLASS profiler" OFF)
option(CUTLASS_ENABLE_TOOLS "Enable CUTLASS tools" OFF)
option(CUTLASS_ENABLE_EXAMPLES "Enable CUTLASS examples" OFF)
option(CUTLASS_ENABLE_CUTE "Enable CuTe support" ON)

# Explicitly disable all kernel operations
set(CUTLASS_LIBRARY_KERNELS "none" CACHE STRING "Disable all CUTLASS kernels")
set(CUTLASS_LIBRARY_OPERATIONS "none" CACHE STRING "Disable all CUTLASS operations")

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUTLASS_NVCC_ARCHS},code=sm_${CUTLASS_NVCC_ARCHS}")

# Add CUTLASS subdirectory
add_subdirectory(cutlass)

add_subdirectory(src)
