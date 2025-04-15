#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <random>
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

// cublas matmul tester (NT row-major inputs, just like in pytorch linear layers)
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void cublas_multiply(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
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
    Tensor a = make_tensor(make_gmem_ptr(A), make_layout(select<0, 2>(shapeMNK), LayoutRight{})); // (M, K)
    Tensor b = make_tensor(make_gmem_ptr(B), make_layout(select<1, 2>(shapeMNK), LayoutRight{})); // (N, K)
    Tensor c = make_tensor(make_gmem_ptr(C), make_layout(select<0, 1>(shapeMNK), LayoutRight{})); // (M, N)

    // Get a tile of A and B for THIS THREAD BLOCK (this is a slice across the K mode)
    // remember cta_tiler is shape (BM, BN, BK)
    // note how both tiles here have the contraction mode BK and the iteration mode k in the same spots
    Tensor gA = local_tile(a, select<0, 2>(cta_tiler), make_coord(blockIdx.x, _)); // (BM, BK, k)
    Tensor gB = local_tile(b, select<1, 2>(cta_tiler), make_coord(blockIdx.y, _)); // (BN, BK, k)
    Tensor gC = local_tile(c, select<0, 1>(cta_tiler), make_coord(blockIdx.x, blockIdx.y)); // BM, BN

    // dim check on gA, B, C
    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<2>(cta_tiler));

    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<2>(cta_tiler));

    CUTE_STATIC_ASSERT_V(size<0>(gC) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(gC) == size<1>(cta_tiler));

    // thread-local subtiles of the global tile:
    // ta is (TM, TK), tb is (TN, TK), tc is its own thing?
    Tensor tAgA = local_partition(gA, ta_tiler, threadIdx.x); // (TM, TK, k) (note that )
    Tensor tBgB = local_partition(gB, tb_tiler, threadIdx.x); // (TN, TK, k)

    CUTE_STATIC_ASSERT_V(shape<1>(ta_tiler) == shape<1>(tb_tiler));

    // alloc smem tensors for a and b
    __shared__ float smem_A[size<0>(cta_tiler) * size<2>(cta_tiler)]; // BM * BK
    __shared__ float smem_B[size<1>(cta_tiler) * size<2>(cta_tiler)]; // BN * BK
    Tensor sA = make_tensor(make_smem_ptr(smem_A), make_layout(select<0, 2>(cta_tiler))); // (BM, BK)
    Tensor sB = make_tensor(make_smem_ptr(smem_B), make_layout(select<1, 2>(cta_tiler))); // (BN, BK)

    CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<2>(cta_tiler));

    CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(sB) == size<2>(cta_tiler));

    // thread-local subtiles of smem, these are just for copying
    Tensor tAsA = local_partition(sA, ta_tiler, threadIdx.x); // however much work done per thread
    Tensor tBsB = local_partition(sB, tb_tiler, threadIdx.x); 


    // if (thread0()) {
    //     printf("tAsA: ");
    //     print(tAsA);
    //     printf("\nta_tiler: ");
    //     print(ta_tiler);
    //     printf("\ntBsB: ");
    //     print(tBsB);
    //     printf("\ntb_tiler: ");
    //     print(tb_tiler);
    //     printf("\n");
    // }


    // don't forget that you ALSO need to partition the smem tiles with C threading, not just the
    // A and B threading for loads

    // get ROWS of sA per thread
    Tensor tCsA = local_partition(sA, tc_tiler, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
    // get COLS of sB per thread
    Tensor tCsB = local_partition(sB, tc_tiler, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
    // the individual rows and cols per thread will each dot-product to have some output tiles
    // those outputs do NOT need to be memory-contiguous
    // this is an implicit partitioning of the output C tile over threads, don't need the actual output C tile
    Tensor tCgC = local_partition(gC, tc_tiler, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

    // thread-local register tensor for accumulating output of C 
    // this implicitly tiles over the output C tile of dim (BM, BN)
    Tensor tCrC = make_tensor_like(tCgC);         

    // Print tensors for thread 0
    if (thread0()) {
        printf("\ntCsA: ");
        print(tCsA);
        printf("\ntCsB: ");
        print(tCsB);
        printf("\ntCgC: ");
        print(tCgC);
        printf("\ntCrC: ");
        print(tCrC);
    }
    clear(tCrC);

    // gemmm mainloop
    for (int k=0; k<size<2>(gA); ++k) {

        // load tiles of a and b
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);
        __syncthreads();

        // does this just work?
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }

    copy(tCrC, tCgC); // remember copy is src to dst
    
}

int main() {
    // M, N, K = 1024, 1024, 1024
    auto gemm_shape = make_shape(256, 512, 1024);
    // auto gemm_shape = make_shape(1024, 1024, 1024);
    auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{}); // (BM, BN, BK)
    auto ta_tiler = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
    auto tb_tiler = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
    auto tc_tiler = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{});

    // auto gemm_shape = make_shape(8, 8, 8);
    // auto cta_tiler = make_shape(Int<8>{}, Int<8>{}, Int<8>{}); // (BM, BN, BK)
    // auto ta_tiler = make_layout(make_shape(Int<32>{}, Int<1>{}));
    // auto tb_tiler = make_layout(make_shape(Int<32>{}, Int<1>{}));
    // auto tc_tiler = make_layout(make_shape(Int<4>{}, Int<8>{}));

    auto grid_shape = select<1, 2>(tiled_divide(make_layout(select<0, 1>(gemm_shape)), select<0, 1>(cta_tiler)));

    thrust::host_vector<float> h_A(size(select<0, 2>(gemm_shape)));
    thrust::host_vector<float> h_B(size(select<1, 2>(gemm_shape)));
    thrust::host_vector<float> h_C(size(select<0, 1>(gemm_shape)));
    thrust::host_vector<float> h_C_cublas(size(select<0, 1>(gemm_shape)));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < size(select<0, 2>(gemm_shape)); i++) {
        h_A[i] = dist(gen);
        // h_A[i] = 1.0f;
        // h_A[i] = i*2;
    }
    for (int i = 0; i < size(select<1, 2>(gemm_shape)); i++) {
        h_B[i] = dist(gen);
        // h_B[i] = 1.0f;
        // h_B[i] = i*2+1;
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


    dim3 gridDim(size<0>(grid_shape), size<1>(grid_shape)); // BMxBN

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
    // for (float& val : h_C_cublas)
    //     val = 3.0;

    // Compare results
    int32_t errors = 0;
    int32_t const kErrorLimit = 10;
    float const kTolerance = 1e-3f;  // Tolerance for floating point comparison

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

