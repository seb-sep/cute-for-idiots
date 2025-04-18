cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cutlass_for_idiots LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Set default CUDA architecture if not specified
# 6.1 is pascal for gtx 1070
#6.0 for p100
#8.0 for a100
if(NOT DEFINED CUTLASS_NVCC_ARCHS)
    set(CUTLASS_NVCC_ARCHS 80 CACHE STRING "CUDA architectures to compile for")
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

# Create your executable
add_executable(tiled_copy tiled_copy.cu)

# Link against minimal CUTLASS components
target_link_libraries(tiled_copy
    PRIVATE
    CUTLASS
    ${CUDA_LIBRARIES}
)

# Include directories - only the essential ones
target_include_directories(tiled_copy
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/cute/include  # CuTe headers
    ${CUDA_INCLUDE_DIRS}
)

# Set C++ standard for the target
set_target_properties(tiled_copy PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Optional: Set up installation
install(TARGETS tiled_copy
    RUNTIME DESTINATION bin
)

add_executable(print_layouts print_layouts.cu)

target_link_libraries(print_layouts
    PRIVATE
    CUTLASS
    ${CUDA_LIBRARIES}
)

target_include_directories(print_layouts
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/cute/include  # CuTe headers
    ${CUDA_INCLUDE_DIRS}
)

# Create executable for simple_gemm
add_executable(simple_gemm simple_gemm.cu)

# Link against CUTLASS and CUDA libraries including cuBLAS
target_link_libraries(simple_gemm
    PRIVATE
    CUTLASS
    ${CUDA_LIBRARIES}
    cublas
)

# Include directories for simple_gemm
target_include_directories(simple_gemm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/cute/include  # CuTe headers
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(simple_gemm PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_executable(simple_gemm_tc simple_gemm_tc.cu)

target_link_libraries(simple_gemm_tc
    PRIVATE
    CUTLASS
    ${CUDA_LIBRARIES}
    cublas
)

target_include_directories(simple_gemm_tc
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/cute/include  
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_BF16_INCLUDE_DIRS}
)

set_target_properties(simple_gemm_tc PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)