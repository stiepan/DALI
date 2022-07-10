// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_CONVOLUTION_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_CONVOLUTION_GPU_H_

#include "cutlass/conv/device/batched_implict_gemm_convolution.h"
#include "cutlass/conv/kernel/fixed_channel_2d_conv.h"
// #include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
// #include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/scratch.h"
// #include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {
namespace kernels {

template <typename Out, typename In, typename W, int axes, bool has_channels = false,
          bool is_sequence = false>
struct Convolution2dGpu;

template <typename Out, typename In, typename W, bool has_channels, bool is_sequence>
struct Convolution2dGpu<Out, In, W, 2, has_channels, is_sequence> {
  static constexpr int axes = 2;
  static constexpr int sequence_axes = static_cast<int>(is_sequence);
  static constexpr int channel_axes = static_cast<int>(has_channels);
  static constexpr int ndim = sequence_axes + axes + channel_axes;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  static_assert(std::is_same<Intermediate, W>::value);

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape,
                           const TensorListShape<2>& window_sizes) {
    KernelRequirements req;
    ScratchpadEstimator se;
    se.add<mm::memory_kind::device, UnderlyingKernelParms>(in_shape.num_samples());
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);
    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, 2>& windows) {
    int num_samples = in.shape.num_samples();

    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    const auto& in_shapes = in.shape;
    const auto& win_shapes = windows.shape;

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_out_shape = in_shapes[sample_idx];
      const auto& win_shape = win_shapes[sample_idx];
      int h_idx = 0;
      int n = 1;
      int c = 1;
      if (is_sequence) {
        n = in_out_shape[0];
        h_idx++;
      }
      int h = in_out_shape[h_idx];
      int w = in_out_shape[h_idx + 1];
      if (has_channels) {
        c = in_out_shape[h_idx + 2];
      }
      int r = win_shape[0];
      int s = win_shape[1];
      cutlass::Tensor4DCoord input_size{n, h, w * c, 1};
      cutlass::Tensor4DCoord filter_size{1, r, s, 1};
      ProblemSizeDesc problem_size(input_size,    // NHWC
                                   filter_size,   // KRSC
                                   {1, 1, c, 1},  // pad_h, _, pad_w, _
                                   {1, 1},        // stride_h, stride_w
                                   {1, c},        // dilation_h, dilation_w
                                   input_size,    // output shape
                                   cutlass::conv::Mode::kCrossCorrelation,
                                   1  // k-split-slices
      );
      auto in_layout = LayoutInputA::packed(input_size);
      auto filter_layout = LayoutInputB::packed(filter_size);
      auto out_layout = LayoutOutput::packed(input_size);
      TensorRefA tensor_a(in.tensor_data(sample_idx), in_layout);
      TensorRefB tensor_b(windows.tensor_data(sample_idx), filter_layout);
      TensorRefC tensor_c(out.tensor_data(sample_idx), out_layout);
      TensorRefC tensor_d(out.tensor_data(sample_idx), out_layout);
      SampleDesc sample_desc{
          problem_size,              // shapes
          tensor_a.non_const_ref(),  // ptr to input tensor (A)
          tensor_b.non_const_ref(),  // ptr to filter tensor (B)
          tensor_c,                  // ptr to addent tensor (C)
          tensor_d,                  // ptr to output tensor (D)
          {1., 0.}  // epillogue scalras (alpha, beta) (D = alpha * conv(A, B) + beta * C)
      };
      samples_desc_.push_back(sample_desc);
    }

    size_t workspace_size = implicit_gemm_op_.get_workspace_size(samples_desc_);
    assert(workspace_size <= sizeof(UnderlyingKernelParms) * num_samples);
    auto* workspace = ctx.scratchpad->AllocateGPU<UnderlyingKernelParms>(num_samples);
    auto status = implicit_gemm_op_.can_implement(samples_desc_);
    DALI_ENFORCE(status == cutlass::Status::kSuccess,
                 make_string("Operation not possible: ", cutlass::cutlassGetStatusString(status)));
    status = implicit_gemm_op_.initialize(samples_desc_, workspace, ctx.gpu.stream);
    DALI_ENFORCE(status == cutlass::Status::kSuccess,
                 make_string("Initialization failed: ", cutlass::cutlassGetStatusString(status)));
    status = implicit_gemm_op_(ctx.gpu.stream);
    DALI_ENFORCE(
        status == cutlass::Status::kSuccess,
        make_string("Launching cutlass op failed: ", cutlass::cutlassGetStatusString(status)));
  }

 private:
  using ElementAccumulator = Intermediate;      // Data type of accumulator
  using ElementComputeEpilogue = Intermediate;  // Data type of epilogue computation (alpha, beta)
  using ElementInputA = In;                     // Data type of elements in input tensor
  using ElementInputB = W;                      // Data type of elements in input tensor
  using ElementOutput = Out;                    // Data type of elements in output tensor

  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using MMAOp = cutlass::arch::OpClassSimt;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm70;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::BatchedIdentityThreadblockSwizzle;

  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;

  // Number of pipelines you want to use
  static constexpr int NumStages = 2;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,            // Data type of output matrix.
      1,                        // The number of elements per vectorized.
                                // memory access. This becomes the vector width of
                                // math instructions in the epilogue too.
      ElementAccumulator,       // Data type of accumulator
      ElementComputeEpilogue>;  // Data type for alpha/beta in linear combination

  using Conv2dFixedChannelsKernel = typename cutlass::conv::kernel::Conv2dFixedChannelFprop<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
      ElementAccumulator, MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
      SwizzleThreadBlock, NumStages, cutlass::arch::OpMultiplyAddSaturate,
      cutlass::conv::IteratorAlgorithm::kFixedChannels, cutlass::conv::StrideSupport::kStrided,
      AlignmentA, AlignmentB>::Kernel;

  using GroupedImplicitFixedChannelsGemm =
      cutlass::conv::device::GroupedImplicitGemm<Conv2dFixedChannelsKernel>;

  using UnderlyingKernelParms = typename Conv2dFixedChannelsKernel::Params;
  using ProblemSizeDesc = typename cutlass::conv::Conv2dProblemSize;
  using SampleDesc = typename Conv2dFixedChannelsKernel::Arguments;
  using TensorRefA = cutlass::TensorRef<const ElementInputA, LayoutInputA>;
  using TensorRefB = cutlass::TensorRef<const ElementInputB, LayoutInputB>;
  using TensorRefC = typename Conv2dFixedChannelsKernel::TensorRefC;

  std::vector<SampleDesc> samples_desc_;
  GroupedImplicitFixedChannelsGemm implicit_gemm_op_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_CONVOLUTION_GPU_H_
