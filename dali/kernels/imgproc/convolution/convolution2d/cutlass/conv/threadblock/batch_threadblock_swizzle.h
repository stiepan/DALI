#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"


namespace cutlass {
namespace gemm {
namespace threadblock {

struct BatchedIdentityThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  BatchedIdentityThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    // the z dim is used for batch size by the batched implicit gemm wrapper
    assert(split_k_slices == 1);

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      1);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    gemm::GemmCoord implicit_gemm_problem_size =
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape) const {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = 0; // hide from the kernel the fact that z is used form sample indexing

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
