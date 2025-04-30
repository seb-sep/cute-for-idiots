#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cuda_bf16.h>  // Add bfloat16 support

using namespace cute;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)



// cublas matmul tester (NT row-major inputs, just like in pytorch linear layers)
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void cublas_multiply(float* A, float* B, float* C, int M, int N, int K, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For NT matmul with row-major inputs:
    // C = A * B^T
    // Since cuBLAS expects column-major, we compute:
    // C^T = (B^T)^T * A^T = B * A^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            B, K,  
                            A, K,  
                            &beta,
                            C, N)); 
    
}

// bfloat16 version of cublas matmul
void cublas_multiply_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, 
                         int M, int N, int K, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For NT matmul with row-major inputs:
    // C = A * B^T
    // Since cuBLAS expects column-major, we compute:
    // C^T = (B^T)^T * A^T = B * A^T
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, CUDA_R_16BF, K,  
                             A, CUDA_R_16BF, K,  
                             &beta,
                             C, CUDA_R_16BF, N,
                             CUDA_R_32F,            // accumulate with FP32 
                             CUBLAS_GEMM_DEFAULT)); 
}


// return average time of runs in ms
template <typename F>
float bench_gemm(F gemm_func) {
    // warmup runs
    for (int i = 0; i < 10; ++i)
        gemm_func();

    int n_runs = 100;
    // timed runs
    thrust::host_vector<float> times(n_runs);
    for (int i = 0; i < n_runs; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        gemm_func();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop); // cuda events record in ms
        times[i] = ms;
        // clear l2 for next run
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-persistence-example
        cudaCtxResetPersistingL2Cache();
    }

    // return average of times
    return (std::accumulate(times.begin(), times.end(), 0.0f)) / n_runs;
}

