#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template <class ProblemShape, class CtaTiler, class SmemLayout>
__global__ void tiled_copy(ProblemShape shapeMN, CtaTiler tiler, float* A, float* B) {
    // instantiate global tensors
    Tensor a = make_tensor(make_gmem_ptr(A), make_layout(shapeMN, LayoutRight{}));
    Tensor b = make_tensor(make_gmem_ptr(b), make_layout(shapeMN, LayoutRight{}));

    __shared__ float copy_S[cosize_v<SmemLayout>]; // cosize_v statically gets the size of a layout
    Tensor shared_tile = make_tensor(make_smem_ptr(copy_S), SmemLayout);

    // tile of A and B
    Tensor a_tiled = local_tile(a, tiler, make_coord(blockIdx.x, blockIdx.y));
    Tensor b_tiled = zipped_divide(a, tiler)(make_coord(_, _), make_coord(blockIdx.x, blockIdx.y)); 

    // these formulations for tiling should be the SAME
    // assert(layout<0>(a_tiled) == layout<0>(b_tiled));
    // assert(layout<1>(a_tiled) == layout<1>(b_tiled));
    // assert(stride<0>(a_tiled) == stride<0>(b_tiled));
    // assert(stride<1>(a_tiled) == stride<1>(b_tiled));

    CUTE_STATIC_ASSERT_V(decltype(shape<0>(a_tiled)) == decltype(shape<0>(b_tiled)));
    CUTE_STATIC_ASSERT_V(decltype(shape<1>(a_tiled)) == decltype(shape<1>(b_tiled)));
    CUTE_STATIC_ASSERT_V(decltype(stride<0>(a_tiled)) == decltype(stride<0>(b_tiled)));
    CUTE_STATIC_ASSERT_V(decltype(stride<1>(a_tiled)) == decltype(stride<1>(b_tiled)));

    // inner mode of tile should have same layout as smem tile
    CUTE_STATIC_ASSERT_V(decltype(shape<0>(a_tiled)) == decltype(shape<0>(shared_tile)));
    CUTE_STATIC_ASSERT_V(decltype(shape<1>(a_tiled)) == decltype(shape<1>(shared_tile)));
    CUTE_STATIC_ASSERT_V(decltype(stride<0>(a_tiled)) == decltype(stride<0>(shared_tile)));
    CUTE_STATIC_ASSERT_V(decltype(stride<1>(a_tiled)) == decltype(stride<1>(shared_tile)));

    copy(a_tiled, shared_tile);
    __syncthreads();
    copy(shared_tile, b_tiled);
}