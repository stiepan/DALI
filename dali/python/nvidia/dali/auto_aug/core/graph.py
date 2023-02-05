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

from nvidia.dali import fn
from nvidia.dali import types

from nvidia.dali.data_node import DataNode as _DataNode

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


def random_operation_idx_choice(num_total_ops, num_levels=1, rng=None):
    shape = tuple() if num_levels == 1 else (num_levels, )
    if rng is None:
        rng = np.random.default_rng(12345)

    def random_choice(_):
        return rng.choice(range(num_total_ops), shape)

    return fn.external_source(source=random_choice, batch=False)


def get_signed_magnitude(magnitudes, randomly_negate, bin_idx):
    magnitude = magnitudes[bin_idx // 2]
    if randomly_negate and bin_idx % 2:
        magnitude = -magnitude
    return np.array(magnitude, dtype=magnitudes.dtype)


def map_random_unsigned_bin(magnitudes, augmentation):
    assert not augmentation.randomly_negate
    return np.array([augmentation.as_param(magnitude) for magnitude in magnitudes])


def map_random_signed_bin(magnitudes, augmentation):
    randomly_negate = augmentation.randomly_negate
    as_param = augmentation.as_param
    return np.array([
        as_param(get_signed_magnitude(magnitudes, randomly_negate, bin_idx))
        for bin_idx in range(len(magnitudes) * 2)
    ])


def map_fixed_signed_bin(fixed_bin_idx):

    def inner(magnitudes, augmentation):
        randomly_negate = augmentation.randomly_negate
        as_param = augmentation.as_param
        magnitudes = magnitudes[fixed_bin_idx:fixed_bin_idx + 2]
        return np.array([
            as_param(get_signed_magnitude(magnitudes, randomly_negate, bin_idx))
            for bin_idx in range(2)
        ])

    return inner


def magnitude_bin(num_bins, use_signed_magnitudes, fixed_bin=None, num_levels=1):
    assert not isinstance(fixed_bin, _DataNode)
    shape = tuple() if num_levels == 1 else (num_levels, )
    if fixed_bin is not None:
        if not use_signed_magnitudes:
            return fixed_bin, None
        else:
            bin_idx = fn.random.uniform(range=[0, 1], dtype=types.INT32, shape=shape)
            return bin_idx, map_fixed_signed_bin(fixed_bin)
    else:
        if not use_signed_magnitudes:
            bin_idx = fn.random.uniform(range=[0, num_bins - 1], dtype=types.INT32, shape=shape)
            return bin_idx, map_random_unsigned_bin
        else:
            num_rand_bins = 2 * num_bins
            bin_idx = fn.random.uniform(range=[0, num_rand_bins - 1], dtype=types.INT32,
                                        shape=shape)
            return bin_idx, map_random_signed_bin


def split_samples_between_ops(op_range_lo, op_range_hi, ops, selected_op_idx, op_kwargs):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](**op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        return split_samples_between_ops(op_range_lo, mid, ops, selected_op_idx, op_kwargs)
    else:
        return split_samples_between_ops(mid + 1, op_range_hi, ops, selected_op_idx, op_kwargs)


def apply_selected_operators(ops, selected_op_idx, op_kwargs):
    return split_samples_between_ops(0, len(ops) - 1, ops, selected_op_idx, op_kwargs)
