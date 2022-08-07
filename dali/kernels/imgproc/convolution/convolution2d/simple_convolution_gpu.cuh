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
  unsigned int h, w, c, vol;
  unsigned int r, s, filter_vol;
  unsigned int in_workspace_width, in_workspace_size;
};

template <typename In>
__host__ __device__ float get_value(const In* in, int in_h, int in_wc, int H, int WC) {
  if (in_h < 0 || in_wc < 0 || in_h >= H || in_wc >= WC) {
    return 0;
  }
  int in_idx = in_h * WC + in_wc;
  return in[in_idx];
}

template <int lanes, typename Out, typename In, typename W>
__global__ void conv2d(const SampleDesc<Out, In, W>* __restrict__ descs) {
  extern __shared__ float shm[];
  int sample_idx = blockIdx.z;
  auto sample_desc = descs[sample_idx];
  auto* global_filter = sample_desc.filter;
  float* filter = shm + sample_desc.in_workspace_size;
  for (int i = threadIdx.x; i < sample_desc.filter_vol; i += blockDim.x) {
    filter[i] = global_filter[i];
  }
  __syncthreads();
  // int grid_size = lanes * gridDim.x * blockDim.x;
  unsigned int rr = sample_desc.r / 2;
  unsigned int rs = sample_desc.s / 2;
  unsigned int wc = sample_desc.w * sample_desc.c;
  auto* out = sample_desc.out;
  auto* in = sample_desc.in;
  // sample_desc.out[idx] = sample_desc.in[idx];
  for (int h_start = lanes * blockIdx.y; h_start < sample_desc.h; h_start += gridDim.y * lanes) {
    for (int w_start = blockDim.x * blockIdx.x; w_start < wc; w_start += gridDim.x * blockDim.x) {
      for (int w = threadIdx.x; w < blockDim.x + (sample_desc.s - 1) * sample_desc.c;
           w += blockDim.x) {
        int global_w = w_start + w - rs * sample_desc.c;
#pragma unroll lanes
        for (int h = 0; h < lanes + sample_desc.r - 1; h++) {
          int global_h = h_start + h - rr;
          shm[h * sample_desc.in_workspace_width + w] =
              get_value(in, global_h, global_w, sample_desc.h, wc);
        }
      }
      __syncthreads();
      int filter_pos = 0;
      float acc[lanes] = {};
      for (int r = 0; r < sample_desc.r; r++) {
        for (int s = 0; s < sample_desc.s; s++) {
          auto filter_coef = filter[filter_pos++];
          int inp_wc = threadIdx.x + s * sample_desc.c;
#pragma unroll
          for (int lane = 0; lane < lanes; lane++) {
            int inp_h = lane + r;
            float in_val = shm[inp_h * sample_desc.in_workspace_width + inp_wc];
            acc[lane] += in_val * filter_coef;
          }
        }
      }
      int in_w = w_start + threadIdx.x;
      if (in_w < wc) {
#pragma unroll
        for (int lane = 0; lane < lanes; lane++) {
          int in_h = h_start + lane;
          if (in_h < sample_desc.h) {
            out[in_h * wc + in_w] = acc[lane];
          }
        }
      }
    }
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

  static constexpr unsigned int block_width = 64;
  static constexpr unsigned int lanes = 8;
  static constexpr unsigned int max_grid_height = 32 * 8;
  static constexpr unsigned int max_grid_width = 32;

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

    unsigned int max_width = 0, max_height = 0, max_total_workspace = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_out_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      unsigned int vol = volume(in_out_shape);
      // max_vol = std::max(max_vol, vol);
      unsigned int h = in_out_shape[0], w = in_out_shape[1];
      unsigned int c = has_channel_dim ? in_out_shape[2] : 1;
      unsigned int r = filter_shape[0], s = filter_shape[1];
      unsigned int filter_vol = volume(filter_shape);
      unsigned int workspace_width = block_width + (s - 1) * c;
      unsigned int workspace_size = workspace_width * (lanes + r - 1);
      max_width = std::max(max_width, w * c);
      max_height = std::max(max_height, h);
      max_total_workspace = std::max(max_total_workspace, filter_vol + workspace_size);
      SampleDesc<Out, In, W> desc = {filters.tensor_data(sample_idx),
                                     in.tensor_data(sample_idx),
                                     out.tensor_data(sample_idx),
                                     h,
                                     w,
                                     c,
                                     vol,
                                     r,
                                     s,
                                     filter_vol,
                                     workspace_width,
                                     workspace_size};
      samples_desc_.push_back(desc);
    }
    SampleDesc<Out, In, W>* descs_dev =
        ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(samples_desc_));
    unsigned int num_blocks_h = (max_height + lanes - 1) / lanes;
    unsigned int num_blocks_w = (max_width + block_width - 1) / block_width;
    num_blocks_h = std::min(num_blocks_h, max_grid_height);
    num_blocks_w = std::min(num_blocks_w, max_grid_width);
    dim3 grid = {num_blocks_w, num_blocks_h, num_samples};
    dim3 block = {block_width, 1, 1};
    conv2d<lanes><<<grid, block, max_total_workspace * sizeof(float), ctx.gpu.stream>>>(descs_dev);
    CUDA_CALL(cudaGetLastError());
  }

 private:
  std::vector<SampleDesc<Out, In, W>> samples_desc_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION2D_SIMPLE_CONVOLUTION_GPU_H_
