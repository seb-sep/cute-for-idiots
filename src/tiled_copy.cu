#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using namespace cute;

/*
    of course you would never make a copy kernel by bouncing through smem, but just for practice
*/
template <class ProblemShape, class CtaTiler, class ThreadTiler>
__global__ void tiled_shared_copy_kernel(ProblemShape shapeMN, CtaTiler cta_tiler, ThreadTiler thread_tiler, float* A, float* B) {
    // instantiate global tensors
    // default layout is layoutLeft (col-major)
    Tensor a = make_tensor(make_gmem_ptr(A), make_layout(shapeMN));
    Tensor b = make_tensor(make_gmem_ptr(B), make_layout(shapeMN));

    __shared__ float smem_A[size(cta_tiler)];
    Tensor sA = make_tensor(make_smem_ptr(smem_A), cta_tiler);

    // tile of A and B
    // the tiler is the shape of the actual inner tile
    Tensor gA = local_tile(a, cta_tiler, make_coord(blockIdx.x, blockIdx.y));
    Tensor gB = local_tile(b, cta_tiler, make_coord(blockIdx.x, blockIdx.y));
    if (thread0()) {
        cute::print(zipped_divide(gA, cta_tiler));
        printf("\n");
        cute::print(gB);
        printf("\n");
    }


    // we defined a 1d threadblock with the size of the thread tiler, so we _only_
    // need to use threadIdx.x here
    // local_partition selects over the outer mode, NOT the inner tile mode like 
    // local_tile does. This way, each threadIdx.x is not given a contiuous tile, instead,
    // each tile is evenly split across all threads, instea of one tile per thread
    Tensor tgA = local_partition(gA, thread_tiler, threadIdx.x);
    Tensor tsA = local_partition(sA, thread_tiler, threadIdx.x);
    Tensor tgB = local_partition(gB, thread_tiler, threadIdx.x);

    if (thread0()) {
        cute::print(zipped_divide(tgA, thread_tiler));
        printf("\n");
        cute::print(tsA);
        printf("\n");
        cute::print(tgB);
        printf("\n");
    }
    copy(tgA, tsA);
    __syncthreads();
    copy(tsA, tgB);
}

int main(int argc, char** argv) {

  //
  // Given a 2D shape, perform an efficient copy
  //

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(128, 64);
  // auto tensor_shape = make_shape(256, 512);

  //
  // Allocate and initialize
  //

  thrust::host_vector<float> h_S(size(tensor_shape));
  thrust::host_vector<float> h_D(size(tensor_shape));

  auto cta_tiler = make_shape(Int<128>{}, Int<64>{});
  auto thread_tiler = make_layout(make_shape(Int<32>{}, Int<4>{}));

  // Tensor tensor_hS = make_tensor(h_S.data(), make_layout(tensor_shape));
  // Tensor tensor_hD = make_tensor(h_D.data(), make_layout(tensor_shape));
  // printf("tensor_hS layout: ");
  // cute::print(layout(tensor_hS));
  // printf("\n");
  // printf("thread_tiler layout: ");
  // cute::print(layout(thread_tiler));
  // printf("\n");

for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = i;
    h_D[i] = 0.0;
  }


  // // Then partition within the block
  // for (size_t i = 0; i < size(thread_tiler); ++i) {
  //   auto tile_s = local_partition(tensor_hS, thread_tiler, i);
  //   auto tile_d = local_partition(tensor_hD, thread_tiler, i);
    
  //   if (i < 2) {  // Print first couple iterations
  //       printf("Thread %zu:\n", i);
  //       printf("Source tile: ");
  //       cute::print(tile_s);
  //       printf("\nDestination tile: ");
  //       cute::print(tile_d);

  //       printf("\nFirst few elements: %f %f %f %f\n", 
  //              float(tile_s(0)), float(tile_s(1)), 
  //              float(tile_s(2)), float(tile_s(3)));
  //   }
    
  //   copy(tile_s, tile_d);
  // }



  // does this copy to device??? sweet
  thrust::device_vector<float> d_S = h_S;
  thrust::device_vector<float> d_D = h_D;

  //
  // Make tensors
  //

  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

  //
  // Tile tensors
  //

  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.

  if ((size<0>(tensor_shape) % size<0>(cta_tiler)) || (size<1>(tensor_shape) % size<1>(cta_tiler))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  // Equivalent check to the above
  if (not evenly_divides(tensor_shape, cta_tiler)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  // tiled_divide has the tile mode first like zipped_divide, but the outer modes are flattened out unlike
  // in zipped_divide where they're collected into a single tuple mode
  Tensor tiled_tensor_S = tiled_divide(tensor_S, cta_tiler);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, cta_tiler);      // ((M, N), m', n')

    // having done a tiled divide lets us easily extract the modes instead of doing nested access crap
   cute::print(shape(tiled_tensor_D));
    printf("gridDim: %d, %d\n", size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));
  fflush(stdout);
  // dim3 gridDim (2, 8);   // Grid shape corresponds to modes m' and n'
  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thread_tiler));

  //
  // Launch the kernel
  //
tiled_shared_copy_kernel<<<gridDim, blockDim>>>(tensor_shape, cta_tiler, thread_tiler, thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()));
fflush(stdout);
  CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
  // cudaError result = cudaDeviceSynchronize();
  // if (result != cudaSuccess) {
  //   std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
  //   return -1;
  // }


  //
  // Verify
  //

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "th error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}
