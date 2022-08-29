// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_2D_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_2D_GPU_H_

#include <limits>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace conv_2d {
template <typename Out_, typename In_, typename W_, typename Acc_, int lanes_>
struct SampleDesc {
  using Acc = Acc_;
  using Out = Out_;
  using In = In_;
  using W = W_;
  static constexpr int lanes = lanes_;

  Out* __restrict__ out;
  const In* __restrict__ in;
  const W* __restrict__ filter;

  struct ShapeDesc {
    int64_t hwc;
    int wc, f, h, w, c;
    int filter_vol, r, s;
    int filter_top_anchor, filter_left_anchor;
    int in_workspace_width, filter_offset;
  } shape;
};


DALI_HOST_DEV DALI_FORCEINLINE int border_reflect_101(int idx, int len) {
  while (true) {
    if (idx < 0) {
      idx = -idx;
    } else if (idx >= len) {
      idx = 2 * len - 2 - idx;
    } else {
      return idx;
    }
  }
}

DALI_HOST_DEV DALI_FORCEINLINE int border_reflect_101_strided(int idx, int reflect_dim_size,
                                                              int inner_stride, int total_stride) {
  if (idx < 0) {
    int reflect_dim_idx = (idx + 1) / inner_stride - 1;
    int inner_dim_idx = (idx + 1) % inner_stride + inner_stride - 1;
    return border_reflect_101(reflect_dim_idx, reflect_dim_size) * inner_stride + inner_dim_idx;
  }
  if (idx >= total_stride) {
    return border_reflect_101(idx / inner_stride, reflect_dim_size) * inner_stride +
           idx % inner_stride;
  }
  return idx;
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const SampleDescT& sample_desc,
                                                    const typename SampleDescT::In* in,
                                                    typename SampleDescT::In* in_workspace,
                                                    int h_start, int w_start) {
  for (int w = threadIdx.x; w < sample_desc.shape.in_workspace_width; w += blockDim.x) {
    int global_w = w_start + w + sample_desc.shape.filter_left_anchor * sample_desc.shape.c;
    global_w = border_reflect_101_strided(global_w, sample_desc.shape.w, sample_desc.shape.c,
                                          sample_desc.shape.wc);
    auto load_row = [&](int h) {
      int global_h = border_reflect_101(h_start + h + sample_desc.shape.filter_top_anchor,
                                        sample_desc.shape.h);
      in_workspace[h * sample_desc.shape.in_workspace_width + w] =
          in[global_h * sample_desc.shape.wc + global_w];
    };
#pragma unroll
    for (int h = 0; h < SampleDescT::lanes; h++) {
      load_row(h);
    }
    for (int h = SampleDescT::lanes; h < SampleDescT::lanes + sample_desc.shape.r - 1; h++) {
      load_row(h);
    }
  }
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE void load_filter_to_shm(const SampleDescT& sample_desc,
                                                     typename SampleDescT::W* filter) {
  auto* global_filter = sample_desc.filter;
  for (int i = threadIdx.x; i < sample_desc.shape.filter_vol; i += blockDim.x) {
    filter[i] = global_filter[i];
  }
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE void store_acc_in_global_output(const SampleDescT& sample_desc,
                                                             typename SampleDescT::Out* out,
                                                             const typename SampleDescT::Acc* acc,
                                                             int h_start, int w_start) {
  int reflect_dim_idx = w_start + threadIdx.x;
  if (reflect_dim_idx < sample_desc.shape.wc) {
#pragma unroll
    for (int lane = 0; lane < SampleDescT::lanes; lane++) {
      int in_h = h_start + lane;
      if (in_h < sample_desc.shape.h) {
        out[in_h * sample_desc.shape.wc + reflect_dim_idx] =
            ConvertSat<typename SampleDescT::Out>(acc[lane]);
      }
    }
  }
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE void shm_input_filter_product(const SampleDescT& sample_desc,
                                                           const typename SampleDescT::W* filter,
                                                           const typename SampleDescT::In* in,
                                                           typename SampleDescT::In* in_workspace,
                                                           typename SampleDescT::Acc* acc,
                                                           int h_start, int w_start) {
  __syncthreads();
  load_input_to_shm(sample_desc, in, in_workspace, h_start, w_start);
  __syncthreads();
  for (int s = 0; s < sample_desc.shape.s; s++) {
    int inp_wc = threadIdx.x + s * sample_desc.shape.c;
    for (int r = 0; r < sample_desc.shape.r; r++) {
      auto filter_coef = filter[r * sample_desc.shape.s + s];
#pragma unroll
      for (int lane = 0; lane < SampleDescT::lanes; lane++) {
        int inp_h = lane + r;
        auto in_val = in_workspace[inp_h * sample_desc.shape.in_workspace_width + inp_wc];
        acc[lane] += in_val * filter_coef;
      }
    }
  }
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE void global_input_filter_product(const SampleDescT& sample_desc,
                                                              const typename SampleDescT::W* filter,
                                                              const typename SampleDescT::In* in,
                                                              typename SampleDescT::Acc* acc,
                                                              int h_start, int w_start) {
  for (int s = 0; s < sample_desc.shape.s; s++) {
    int global_w =
        w_start + threadIdx.x + (sample_desc.shape.filter_left_anchor + s) * sample_desc.shape.c;
    global_w = border_reflect_101_strided(global_w, sample_desc.shape.w, sample_desc.shape.c,
                                          sample_desc.shape.wc);
    for (int r = 0; r < sample_desc.shape.r; r++) {
      auto filter_coef = filter[r * sample_desc.shape.s + s];
      // Even without shm, using `lanes` speeds up the kernel by reducing
      // the cost of nested loops arithmetic per single output value
#pragma unroll
      for (int lane = 0; lane < SampleDescT::lanes; lane++) {
        int global_h = h_start + lane + r + sample_desc.shape.filter_top_anchor;
        global_h = border_reflect_101(global_h, sample_desc.shape.h);
        auto in_val = in[global_h * sample_desc.shape.wc + global_w];
        acc[lane] += in_val * filter_coef;
      }
    }
  }
}

template <typename SampleDescT, typename ConvF>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(ConvF&& convf, const SampleDescT& sample_desc) {
  constexpr int lanes = SampleDescT::lanes;
  const auto* in = sample_desc.in;
  auto* out = sample_desc.out;
  for (int f = 0; f < sample_desc.shape.f;
       f++, in += sample_desc.shape.hwc, out += sample_desc.shape.hwc) {
    for (int h_start = lanes * blockIdx.y; h_start < sample_desc.shape.h;
         h_start += gridDim.y * lanes) {
      for (int w_start = blockDim.x * blockIdx.x; w_start < sample_desc.shape.wc;
           w_start += gridDim.x * blockDim.x) {
        float acc[lanes] = {};
        convf(sample_desc, in, acc, h_start, w_start);
        store_acc_in_global_output(sample_desc, out, acc, h_start, w_start);
      }
    }
  }
}

template <typename SampleDescT>
__global__ void conv2d(const SampleDescT* descs) {
  using In = typename SampleDescT::In;
  using W = typename SampleDescT::W;
  using Acc = typename SampleDescT::Acc;

  extern __shared__ char shm[];
  auto sample_desc = descs[blockIdx.z];
  In* in_workspace = reinterpret_cast<In*>(shm);
  W* filter = reinterpret_cast<W*>(shm + sample_desc.shape.filter_offset);
  load_filter_to_shm(sample_desc, filter);
  __syncthreads();
  if (sample_desc.shape.filter_offset) {
    stride_grid(
        [&filter, &in_workspace](const SampleDescT& sample_desc, const In* in, Acc* acc,
                                 int h_start, int w_start) {
          shm_input_filter_product(sample_desc, filter, in, in_workspace, acc, h_start, w_start);
        },
        sample_desc);
  } else {
    stride_grid(
        [&filter](const SampleDescT& sample_desc, const In* in, Acc* acc, int h_start,
                  int w_start) {
          global_input_filter_product(sample_desc, filter, in, acc, h_start, w_start);
        },
        sample_desc);
  }
}
}  // namespace conv_2d

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
struct Convolution2dGpu {
  /* In fact, it computes a corellation not a convolution.
  Flip filter in both dimensions for actual convolution. */

  static constexpr int axes = 2;
  static constexpr int num_sequence_dim = static_cast<int>(has_sequence_dim);
  static constexpr int num_channels_dim = static_cast<int>(has_channel_dim);
  static constexpr int ndim = num_sequence_dim + axes + num_channels_dim;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());

  static constexpr int block_width = 128;
  static constexpr int lanes = 8;
  static constexpr int max_grid_height = 32;
  static constexpr int max_grid_width = 32;
  static constexpr int max_grid_rows = max_grid_height * lanes;
  static constexpr int max_grid_cols = max_grid_width * block_width;
  static constexpr int max_sample_height =
      std::numeric_limits<int>::max() / max_grid_rows * max_grid_rows;
  static constexpr int max_sample_width =
      std::numeric_limits<int>::max() / max_grid_cols * max_grid_cols;

  using SampleDescT = conv_2d::SampleDesc<Out, In, W, Intermediate, lanes>;

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters) {
    auto num_samples = in.shape.num_samples();

    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;

    int shared_mem_limit = GetSharedMemPerBlock();
    int max_width = 0, max_height = 0, max_total_workspace = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_out_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      int required_workspace;
      auto shape_desc = SetupSampleDesc(required_workspace, sample_idx, in_out_shape, filter_shape,
                                        shared_mem_limit);
      max_height = std::max(max_height, shape_desc.h);
      max_width = std::max(max_width, shape_desc.wc);
      max_total_workspace = std::max(max_total_workspace, required_workspace);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc});
    }
    SampleDescT* descs_dev;
    std::tie(descs_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    int num_blocks_h = div_ceil(max_height, lanes);
    int num_blocks_w = div_ceil(max_width, block_width);
    num_blocks_h = std::min(num_blocks_h, max_grid_height);
    num_blocks_w = std::min(num_blocks_w, max_grid_width);
    dim3 grid(num_blocks_w, num_blocks_h, num_samples);
    dim3 block(block_width, 1, 1);
    conv_2d::conv2d<<<grid, block, max_total_workspace, ctx.gpu.stream>>>(descs_dev);
    CUDA_CALL(cudaGetLastError());
  }

 protected:
  template <typename InShape, typename FilterShape>
  typename SampleDescT::ShapeDesc SetupSampleDesc(int& required_worskapce, int sample_idx,
                                                  const InShape& in_out_shape,
                                                  const FilterShape& filter_shape,
                                                  int shared_mem_limit) {
    auto filter_vol = volume(filter_shape);
    auto filter_size = filter_vol * sizeof(W);
    DALI_ENFORCE(
        filter_size <= shared_mem_limit,
        make_string("Filter volume for sample of idx ", sample_idx,
                    " exceedes maximal space available for CUDA kernel. Got filter of size: ",
                    filter_size, "."));
    int r = filter_shape[0], s = filter_shape[1];
    int filter_top_anchor = -r / 2, filter_left_anchor = -s / 2;
    auto f = has_sequence_dim ? in_out_shape[0] : 1;
    auto h = in_out_shape[num_sequence_dim];
    auto w = in_out_shape[num_sequence_dim + 1];
    auto c = has_channel_dim ? in_out_shape[num_sequence_dim + 2] : 1;
    auto wc = w * c;
    auto hwc = h * wc;
    ValidateSampleNumericLimits(sample_idx, r, s, filter_top_anchor, filter_left_anchor, f, h, wc,
                                c);
    int64_t input_workspace_width = block_width + (s - 1) * c;
    int64_t input_workspace_num_elements = input_workspace_width * (lanes + r - 1);
    int64_t filter_offset = align_up(input_workspace_num_elements * sizeof(In), sizeof(W));
    int64_t total_workspace_size = filter_offset + filter_size;
    if (c > block_width || total_workspace_size > shared_mem_limit) {
      filter_offset = input_workspace_width = 0;
      total_workspace_size = filter_size;
    }
    required_worskapce = total_workspace_size;
    return {hwc,
            static_cast<int>(wc),
            static_cast<int>(f),
            static_cast<int>(h),
            static_cast<int>(w),
            static_cast<int>(c),
            static_cast<int>(filter_vol),
            r,
            s,
            filter_top_anchor,
            filter_left_anchor,
            static_cast<int>(input_workspace_width),
            static_cast<int>(filter_offset)};
  }

  template <typename FilterExtent, typename SampleExtent>
  void ValidateSampleNumericLimits(int sample_idx, FilterExtent r, FilterExtent s,
                                   FilterExtent filter_top_anchor, FilterExtent filter_left_anchor,
                                   SampleExtent f, SampleExtent h, SampleExtent wc,
                                   SampleExtent c) {
    DALI_ENFORCE(
        f <= std::numeric_limits<int>::max(),
        make_string("Number of frames for sample of idx ", sample_idx, " exceedes the limit of ",
                    std::numeric_limits<int>::max(), ". Got: ", f, "."));
    DALI_ENFORCE(h <= max_sample_height,
                 make_string("The height of sample of idx ", sample_idx, " exceedes the limit of ",
                             max_sample_height, ". Got: ", h, "."));
    DALI_ENFORCE(wc <= max_sample_width,
                 make_string("The total width and number of channels in sample of idx ", sample_idx,
                             " exceedes the limit of ", max_sample_width, ". Got: ", wc, "."));
    auto height_radious = h + r + filter_top_anchor - 2;
    DALI_ENFORCE(
        height_radious <= std::numeric_limits<int>::max(),
        make_string("The combined height of the sample and filter radious for sample of idx ",
                    sample_idx, " exceedes the limit of ", std::numeric_limits<int>::max(),
                    ". Got: ", height_radious, "."));
    auto width_radious = wc - 1 + (s - 1 + filter_left_anchor) * c;
    DALI_ENFORCE(
        width_radious <= std::numeric_limits<int>::max(),
        make_string("The combined width, number of channels and filter radious for sample of idx ",
                    sample_idx, " exceedes the limit of ", std::numeric_limits<int>::max(),
                    ". Got: ", width_radious, "."));
  }

  std::vector<SampleDescT> samples_desc_;
};


// WAR c++14 odr usage issue (make_string in error message takes them as l-values)
// it should be unnecessary in c++17
template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Convolution2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_sample_height;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Convolution2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_sample_width;


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CONVOLUTION_2D_GPU_H_
