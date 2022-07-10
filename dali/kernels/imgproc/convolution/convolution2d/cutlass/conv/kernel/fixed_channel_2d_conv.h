#pragma once

#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h"

#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_fixed_channels.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_few_channels.h"

namespace cutlass {
namespace conv {
namespace kernel {

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kStrided,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value
> struct Conv2dFixedChannelFprop {

    static_assert(AlignmentA == 1);
    static_assert(AlignmentB == 1);
    static_assert(Stages == 2);
    static_assert(std::is_same<OperatorClass, arch::OpClassSimt>::value);
    static_assert(IteratorAlgorithm == IteratorAlgorithm::kFixedChannels);
    // // we're using blockIdx.z to parallelize samples computation,
    // // so make sure that grid's last dim is
    // // not used for other purposes
    // // TODO(ktokarski) Try to make encode it somehow or whatever and not enforce this here
    // static_assert(IsGemmIdentityThreadblockSwizzle<ThreadblockSwizzle>::value);

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;

  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFixedChannels<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFixedChannels<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};


} // namespace kernel
} // namespace conv
} // namespace cutlass
