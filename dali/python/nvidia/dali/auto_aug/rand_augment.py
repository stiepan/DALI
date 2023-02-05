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

from nvidia.dali.data_node import DataNode as _DataNode

import augmentations as aug
from core.graph import magnitude_bin, random_operation_idx_choice, apply_selected_operators

rand_augment_ops = {
    "shear_x": aug.shear_x.augmentation((0, 0.3), randomly_negate=True),
    "shear_y": aug.shear_y.augmentation((0, 0.3), randomly_negate=True),
    "translate_x": aug.translate_x.augmentation((0, 0.45), randomly_negate=True),
    "translate_y": aug.translate_y.augmentation((0, 0.45), randomly_negate=True),
    "rotate": aug.rotate.augmentation((0, 30), randomly_negate=True),
    "brightness": aug.brightness.augmentation((0, 0.9), aug.shift_enhance_range,
                                              randomly_negate=True),
    "contrast": aug.contrast.augmentation((0, 0.9), aug.shift_enhance_range, randomly_negate=True),
    "color": aug.color.augmentation((0, 0.9), aug.shift_enhance_range, randomly_negate=True),
    "sharpness": aug.sharpness.augmentation((0, 0.9), aug.sharpness_kernel, randomly_negate=True),
    "posterize": aug.posterize.augmentation((0, 4), aug.poster_mask),
    "solarize": aug.solarize.augmentation((256, 0)),
    "invert": aug.invert,
    "equalize": aug.equalize,
    "auto_contrast": aug.auto_contrast
}


def rand_augment(samples, n, m, num_bins=31, shapes=None, max_translate_width=250,
                 max_translate_height=250):
    ops = dict(**rand_augment_ops)
    extra_op_kwargs = {}
    if shapes is not None:
        if isinstance(shapes, _DataNode):
            raise Exception(
                f"The `shapes` parameter must be a node of DALI graph (DataNode), got {shapes}.")
        extra_op_kwargs["shapes"] = extra_op_kwargs
    else:
        ops["translate_x"] = aug.translate_x_no_shape.augmentation((0, max_translate_width)),
        ops["translate_y"] = aug.translate_y_no_shape.augmentation((0, max_translate_height)),
    return apply_rand_augment(rand_augment_ops, samples, n, m, num_bins,
                              extra_op_kwargs=extra_op_kwargs)


def apply_rand_augment(ops, samples, n, m, num_bins, extra_op_kwargs=None):
    if len(ops) == 0:
        return samples
    use_signed_magnitudes = any(op.randomly_negate for op in ops)
    bin_idx, mag_to_param_range = magnitude_bin(num_bins, use_signed_magnitudes, fixed_bin=m,
                                                num_levels=n)
    op_common_kwargs = {
        "num_bins": num_bins,
        "extra_op_kwargs": extra_op_kwargs or {},
        "mag_to_param_range": mag_to_param_range,
    }
    op_idx = random_operation_idx_choice(len(ops), n)
    for level_idx in range(n):
        level_bin_idx = bin_idx if not use_signed_magnitudes else bin_idx[level_idx]
        op_kwargs = dict(samples=samples, bin_idx=level_bin_idx, **op_common_kwargs)
        level_op_idx = op_idx if n == 1 else op_idx[level_idx]
        samples = apply_selected_operators(ops, level_op_idx, op_kwargs)
    return samples
