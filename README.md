# CuTe GEMM

## Layout
`main.cu`: a simple test file taken from the CUTLASS repository for testing a correct build setup.
`tiled_copy.cu`: a CuTe copy kernel for practicing layouts, tile management, and copying.
`simple_gemm.cu`: a naive matmul kernel using CuTe.
`utils.cuh`: Benchamrking and cuBLAS utilities

## Build

Clone the CUTLASS git submodule with `git submodule update --init --recursive`.

Then, select a compute capability in the `CMakeLists.txt` for your GPU:
```cmake
if(NOT DEFINED CUTLASS_NVCC_ARCHS)
    # 80 is for a100
    set(CUTLASS_NVCC_ARCHS 80 CACHE STRING "CUDA architectures to compile for")
endif()
```
Change the number after `CUTLASS_NVCC_ARCHS` to do so, and see the `CMakeLists.txt` for some other options.

Then, build the project with Cmake:
```bash
mkdir build
cd build
cmake ..
make
```

The GEMM kernel and benchmarking is run with 
```bash
cd build
./simple_gemm
```


