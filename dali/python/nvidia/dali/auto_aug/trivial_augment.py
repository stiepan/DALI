# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from core.graph import magnitude_bin, random_operation_idx_choice, apply_selected_operators


def apply_trivial_augment(ops, samples, num_bins=31, extra_op_kwargs=None):
    if len(ops) == 0:
        return samples
    use_signed_magnitudes = any(op.randomly_negate for op in ops)
    bin_idx, mag_to_param_range = magnitude_bin(num_bins, use_signed_magnitudes)
    op_idx = random_operation_idx_choice(len(ops))
    op_kwargs = {
        "samples": samples,
        "bin_idx": bin_idx,
        "mag_to_param_range": mag_to_param_range,
        "num_bins": num_bins,
        "extra_op_kwargs": extra_op_kwargs or {}
    }
    samples = apply_selected_operators(ops, op_idx, op_kwargs)
    return samples
