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
template <class ProblemShape, class CtaTiler>
__global__ void tiled_shared_copy_kernel(ProblemShape shapeMN, CtaTiler tiler, float* A, float* B) {
    // instantiate global tensors
    printf("hello, world\n");
    Tensor a = make_tensor(make_gmem_ptr(A), make_layout(shapeMN, LayoutRight{}));
    Tensor b = make_tensor(make_gmem_ptr(B), make_layout(shapeMN, LayoutRight{}));

    __shared__ float copy_S[size(tiler)]; // cosize_v statically gets the size of a layout
    Tensor shared_tile = make_tensor(make_smem_ptr(copy_S), tiler);

    // tile of A and B
    // the tiler is the shape of the actual inner tile
    Tensor a_tiled = local_tile(a, tiler, make_coord(blockIdx.y, blockIdx.x));
    Tensor b_tiled = local_tile(b, tiler, make_coord(blockIdx.y, blockIdx.x));
        printf("a_tiled: ");
        cute::print(shape(a_tiled));
        printf("b_tiled: ");
        cute::print(shape(b_tiled));
    // Tensor b_tiled = zipped_divide(a, tiler)(make_coord(_, _), make_coord(blockIdx.x, blockIdx.y)); 

    // these formulations for tiling should be the SAME
    // assert(layout<0>(a_tiled) == layout<0>(b_tiled));
    // assert(layout<1>(a_tiled) == layout<1>(b_tiled));
    // assert(stride<0>(a_tiled) == stride<0>(b_tiled));
    // assert(stride<1>(a_tiled) == stride<1>(b_tiled));

    // CUTE_STATIC_ASSERT_V((shape<0>(a_tiled)) == (shape<0>(b_tiled)));
    // CUTE_STATIC_ASSERT_V((shape<1>(a_tiled)) == (shape<1>(b_tiled)));
    // CUTE_STATIC_ASSERT_V((stride<0>(a_tiled)) == (stride<0>(b_tiled)));
    // CUTE_STATIC_ASSERT_V((stride<1>(a_tiled)) == (stride<1>(b_tiled)));

    // // inner mode of tile should have same layout as smem tile
    // CUTE_STATIC_ASSERT_V((shape<0>(a_tiled)) == (shape<0>(shared_tile)));
    // CUTE_STATIC_ASSERT_V((shape<1>(a_tiled)) == (shape<1>(shared_tile)));
    // CUTE_STATIC_ASSERT_V((stride<0>(a_tiled)) == (stride<0>(shared_tile)));
    // CUTE_STATIC_ASSERT_V((stride<1>(a_tiled)) == (stride<1>(shared_tile)));

    copy(a_tiled, shared_tile);
    __syncthreads();
    copy(shared_tile, b_tiled);
}

int main(int argc, char** argv) {

  //
  // Given a 2D shape, perform an efficient copy
  //

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(256, 512);

  //
  // Allocate and initialize
  //

  thrust::host_vector<float> h_S(size(tensor_shape));
  thrust::host_vector<float> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = i;
    h_D[i] = 0.0;
  }

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
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  // Equivalent check to the above
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  // tiled_divide has the tile mode first like zipped_divide, but the outer modes are flattened out unlike
  // in zipped_divide where they're collected into a single tuple mode
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

    // having done a tiled divide lets us easily extract the modes instead of doing nested access crap
    cute::print(shape(tiled_tensor_D));
  dim3 gridDim (2, 8);   // Grid shape corresponds to modes m' and n'
//   dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(128);

  //
  // Launch the kernel
  //
tiled_shared_copy_kernel<<<gridDim, blockDim>>>(tensor_shape, block_shape, thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()));
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
