#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>

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

using namespace cute;

// Function to perform cuBLAS multiplication
void cublas_multiply(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: cuBLAS expects column-major matrices, so we need to transpose the operation
    // C = A * B becomes C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                            N, M, K,
                            &alpha,
                            B, K,  // B^T
                            A, K,  // A^T
                            &beta,
                            C, N));  // C^T
    
    CUBLAS_CHECK(cublasDestroy(handle));
}

template <class ProblemShape, class CtaTiler, 
    class TATiler, class TBTiler, class TCTiler>
__global__ void simple_gemm_kernel(
    ProblemShape shapeMNK, CtaTiler cta_tiler, 
    TATiler ta_tiler, TBTiler tb_tiler, TCTiler tc_tiler,
    float* A, float* B, float* C
) {
    // note that in this kernel (blockIdx.x, blockIdx.y) maps to (M, N)
    // instantiate global tensors (shapeMNK is M N K)
    Tensor a = make_tensor(make_gmem_ptr(A), make_layout(select<0,2>(shapeMNK))); // (M, K)
    Tensor b = make_tensor(make_gmem_ptr(B), make_layout(select<1,2>(shapeMNK))); // (N, K)
    Tensor c = make_tensor(make_gmem_ptr(C), make_layout(select<0,1>(shapeMNK))); // (M, N)

    // Get a tile of A and B for THIS THREAD BLOCK (this is a slice across the K mode)
    // remember cta_tiler is shape (BM, BN, BK)
    // note how both tiles here have the contraction mode BK and the iteration mode k in the same spots
    Tensor gA = local_tile(a, select<0, 2>(cta_tiler), make_coord(blockIdx.x, _)); // (BM, BK, k)
    Tensor gB = local_tile(b, select<1, 2>(cta_tiler), make_coord(blockIdx.y, _)); // (BN, BK, k)

    // alloc smem tensors for a and b
    __shared__ float smem_A[size(select<0, 2>(cta_tiler))];
    __shared__ float smem_B[size(select<1, 2>(cta_tiler))];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), make_layout(select<0, 2>(cta_tiler)));
    Tensor sB = make_tensor(make_smem_ptr(smem_B), make_layout(select<1, 2>(cta_tiler)));

    
    
}

int main() {
    // M, N, K = 1024, 1024, 1024
    auto gemm_shape = make_shape(1024, 1024, 1024);
    auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto ta_tiler = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tb_tiler = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tc_tiler = make_layout(make_shape(Int<16>{}, Int<16>{}));

    thrust::host_vector<float> h_A(size(select<0, 2>(gemm_shape)));
    thrust::host_vector<float> h_B(size(select<1, 2>(gemm_shape)));
    thrust::host_vector<float> h_C(size(select<0, 1>(gemm_shape)));
    thrust::host_vector<float> h_C_cublas(size(select<0, 1>(gemm_shape)));

    for (int i = 0; i < size(select<0, 2>(gemm_shape)); i++) {
        h_A[i] = i;
    }
    for (int i = 0; i < size(select<1, 2>(gemm_shape)); i++) {
        h_B[i] = i;
    }
    for (int i = 0; i < size(select<0, 1>(gemm_shape)); i++) {
        h_C[i] = 0.0;
        h_C_cublas[i] = 0.0;
    }

    // copy to device
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;
    thrust::device_vector<float> d_C_cublas = h_C_cublas;

    dim3 blockDim(size(tc_tiler)); // 16x16
    dim3 gridDim(size<0>(cta_tiler), size<1>(cta_tiler)); // BMxBN

    // Run CUTLASS GEMM
    simple_gemm_kernel<<<gridDim, blockDim>>>(gemm_shape, cta_tiler, ta_tiler, tb_tiler, tc_tiler, 
                                             thrust::raw_pointer_cast(d_A.data()), 
                                             thrust::raw_pointer_cast(d_B.data()), 
                                             thrust::raw_pointer_cast(d_C.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Run cuBLAS GEMM
    cublas_multiply(thrust::raw_pointer_cast(d_A.data()),
                   thrust::raw_pointer_cast(d_B.data()),
                   thrust::raw_pointer_cast(d_C_cublas.data()),
                   size<0>(gemm_shape), size<1>(gemm_shape), size<2>(gemm_shape));

    // Copy results back to host
    h_C = d_C;
    h_C_cublas = d_C_cublas;
    // Compare results
    int32_t errors = 0;
    int32_t const kErrorLimit = 10;
    float const kTolerance = 1e-5f;  // Tolerance for floating point comparison

    for (size_t i = 0; i < size(select<0, 1>(gemm_shape)); ++i) {
        if (std::abs(h_C[i] - h_C_cublas[i]) > kTolerance) {
            std::cerr << "Error. CUTLASS[" << i << "]: " << h_C[i] 
                      << ",   cuBLAS[" << i << "]: " << h_C_cublas[i] << std::endl;

            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "th error." << std::endl;
                return -1;
            }
        }
    }

    if (errors == 0) {
        std::cout << "Success. CUTLASS and cuBLAS results match." << std::endl;
    } else {
        std::cerr << "Found " << errors << " errors." << std::endl;
    }

    return 0;
}

