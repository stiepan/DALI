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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(experimental__Filter)
    .DocStr(R"code(Convolves the image with a provided filter.

The operator accepts exactly two positional arguments:
the batch of images and the batch of filters.

Operator supports 2D images with both channels-first and channels-last layouts.

Filter must be a 2D array of filter coefficients or a sequence of 2D arrays to be applied
frame-wise to a video input.

.. note::
  In fact, the operator computes a correlation, not a convolution,
  i.e. the order of filter elements is not mirrored when computing product of
  filter and a part of an image .

)code")
    .NumInput(2)
    .NumOutput(1)
    .AllowSequences()
    .AddOptionalArg("fill_value",
                    R"code(If specified, input image is padded with
    the provided value, so that the output image has the same shape as input.
    If not specified, border mode 101 is used, i.e. image is reflected around
    the perimeter to fill missing values.)code",
                    std::vector<float>{0}, true, true)
    .AddOptionalArg("anchor",
                    R"code(2D point lying within the filter specifying the placement of the
    filter over an image. The ordering of extents corresponds to the ordering of filter's extents.
    If -1 (the default) is specified for the extent, the middle of that extent is used.)code",
                    std::vector<int>{-1}, true, true)
    .AddOptionalTypeArg("dtype", R"code(Output data type.
Supported type: `FLOAT`. If not set, the input type is used.)code")
    .InputLayout(0, {"FHWC", "FCHW", "HWC", "CHW", "HW"});

}  // namespace dali
