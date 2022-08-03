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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_SIMPLE_CONVOLUTION_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_SIMPLE_CONVOLUTION_GPU_H_

// #include "cutlass/conv/device/batched_implict_gemm_convolution.h"
// #include "cutlass/conv/kernel/fixed_channel_2d_conv.h"
// #include "dali/core/span.h"
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

template <typename Out, typename In, typename W>
struct SampleDesc {
  const W* __restrict__ filter;
  const In* __restrict__ in;
  Out* out;
  int h, w, c, vol;
  int r, s;
};

template <typename Out, typename In, typename W>
__host__ __device__ void position(int idx, const SampleDesc<Out, In, W>& desc, int& h, int& w,
                                  int& c) {
  c = idx % desc.c;
  idx /= desc.c;
  w = idx % desc.w;
  idx /= desc.w;
  h = idx;
}

template <typename In>
__host__ __device__ float get_value(const In* in, int in_h, int in_w, int c, int H, int W, int WC,
                                    int C) {
  if (in_h < 0 || in_w < 0 || in_h >= H || in_w >= W) {
    return 0;
  }
  int in_idx = in_h * WC + in_w * C + c;
  return in[in_idx];
}

template <typename Out, typename In, typename W>
__global__ void conv2d(const SampleDesc<Out, In, W>* __restrict__ descs) {
  int sample_idx = blockIdx.z;
  auto sample_desc = descs[sample_idx];
  auto* filter = sample_desc.filter;
  // for (int i = threadIdx.x; i < sample_desc.filter_vol; i += blockDim.x) {
  //   filter[i] = global_filter[i];
  // }
  // __syncthreads();
  int grid_size = gridDim.x * blockDim.x;
  int rr = sample_desc.r / 2;
  int rs = sample_desc.s / 2;
  int wc = sample_desc.w * sample_desc.c;
  auto* out = sample_desc.out;
  auto* in = sample_desc.in;
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sample_desc.vol; idx += grid_size) {
    // sample_desc.out[idx] = sample_desc.in[idx];
    int h, w, c;
    position(idx, sample_desc, h, w, c);
    int filter_pos = 0;
    float acc = 0;
    for (int r = -rr; r <= rr; r++) {
      int inp_h = h + r;
      for (int s = -rs; s <= rs; s++) {
        int inp_w = w + s;
        float in_val =
            get_value(in, inp_h, inp_w, c, sample_desc.h, sample_desc.w, wc, sample_desc.c);
        acc += in_val * filter[filter_pos++];
      }
    }
    out[idx] = acc;
  }
}

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
struct Convolution2dGpu {
  static constexpr int axes = 2;
  static constexpr int num_sequence_dim = static_cast<int>(has_sequence_dim);
  static constexpr int num_channels_dim = static_cast<int>(has_channel_dim);
  static constexpr int ndim = num_sequence_dim + axes + num_channels_dim;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  static_assert(std::is_same<Intermediate, W>::value);

  KernelRequirements Setup(KernelContext& ctx, const TensorListShape<ndim>& in_shape) {
    KernelRequirements req;
    ScratchpadEstimator se;
    se.add<mm::memory_kind::device, SampleDesc<Out, In, W>>(in_shape.num_samples());
    req.scratch_sizes = se.sizes;
    req.output_shapes.push_back(in_shape);
    return req;
  }

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters) {
    unsigned int num_samples = in.shape.num_samples();

    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;

    int max_vol = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_out_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      int vol = volume(in_out_shape);
      max_vol = std::max(max_vol, vol);
      int h = in_out_shape[0], w = in_out_shape[1];
      int c = has_channel_dim ? in_out_shape[2] : 1;
      int r = filter_shape[0], s = filter_shape[1];
      SampleDesc<Out, In, W> desc = {filters.tensor_data(sample_idx),
                                     in.tensor_data(sample_idx),
                                     out.tensor_data(sample_idx),
                                     h,
                                     w,
                                     c,
                                     vol,
                                     r,
                                     s};
      samples_desc_.push_back(desc);
    }
    SampleDesc<Out, In, W>* descs_dev =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(samples_desc_));
    unsigned int block_size = 128;
    unsigned int num_blocks = ((max_vol + block_size - 1) / block_size);
    dim3 grid = {num_blocks, 1, num_samples};
    dim3 block = {block_size, 1, 1};
    conv2d<<<grid, block, 0, ctx.gpu.stream>>>(descs_dev);
    CUDA_CALL(cudaGetLastError());
  }

 private:
  std::vector<SampleDesc<Out, In, W>> samples_desc_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_SIMPLE_CONVOLUTION_GPU_H_
