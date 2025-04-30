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