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
from nvidia.dali.auto_aug import augmentations as aug
from nvidia.dali.auto_aug.core.utils import operation_idx_random_choice, apply_operators_by_idx, fixed_signed_bin_to_magnitudes

#todo add shapes to shear?
#todo describe the shape param that must take just width, height tuple
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
    "posterize": aug.posterize.augmentation((0, 7), aug.poster_mask_uint8),
    # solarization strength increases with decreasing magnitude (threshold)
    "solarize": aug.solarize.augmentation((256, 0)),
    "solarize_add": aug.solarize_add.augmentation((0, 110)),
    "invert": aug.invert,
    "equalize": aug.equalize,
    "auto_contrast": aug.auto_contrast,
    "identity": aug.identity,
}

non_monotonic_sharpness = aug.sharpness.augmentation(
    (0.1, 1.9), as_param=aug.sharpness_kernel_shifted, randomly_negate=False)
non_monotonic_ops = {
    "posterize": aug.posterize.augmentation((7, 0), as_param=aug.poster_mask_uint8),
    "solarize": aug.solarize.augmentation((0, 256), as_param=None),
    "brightness": aug.brightness.augmentation((0.1, 1.9), as_param=None, randomly_negate=False),
    "contrast": aug.contrast.augmentation((0.1, 1.9), as_param=None, randomly_negate=False),
    "color": aug.color.augmentation((0.1, 1.9), as_param=None, randomly_negate=False),
    "sharpness": non_monotonic_sharpness,
}

rand_augment_suite = ("shear_x", "shear_y", "translate_x", "translate_y", "rotate", "brightness",
                      "contrast", "color", "sharpness", "posterize", "solarize", "solarize_add",
                      "invert", "equalize", "auto_contrast", "identity")


def rand_augment(samples, n, m, num_magnitude_bins=31, shapes=None, max_translate_width=250,
                 max_translate_height=250, seed=None, monotonic_mag=True, excluded_ops=None):
    ops = dict(**rand_augment_ops)
    extra_op_kwargs = {}
    if shapes is not None:
        if not isinstance(shapes, _DataNode):
            raise Exception(
                f"The `shapes` parameter must be an output of DALI operator (DataNode) that "
                f"describes height and width of the samples node of DALI graph , got {shapes}.")
        extra_op_kwargs["shapes"] = shapes
    else:
        ops["translate_x"] = aug.translate_x_no_shape.augmentation((0, max_translate_width))
        ops["translate_y"] = aug.translate_y_no_shape.augmentation((0, max_translate_height))
    if not monotonic_mag:
        ops.update(non_monotonic_ops)
    excluded_ops = excluded_ops or tuple()
    selected_ops = [ops[name] for name in rand_augment_suite if name not in excluded_ops]
    return apply_rand_augment(selected_ops, samples, n, m, num_magnitude_bins=num_magnitude_bins,
                              seed=seed, extra_op_kwargs=extra_op_kwargs)


def apply_rand_augment(ops, samples, n, m, num_magnitude_bins, seed, extra_op_kwargs=None):
    if m >= num_magnitude_bins:
        raise Exception(
            f"The magnitude `m` must be an integer within `[0, num_magnitude_bins - 1]` range. "
            f"Got `m={m}`, while the `num_magnitude_bins={num_magnitude_bins}`"
        )
    if len(ops) == 0:
        return samples
    use_signed_magnitudes = any(op.randomly_negate for op in ops)
    if not use_signed_magnitudes:
        bin_idx, bins_to_magnitudes_map = m, None
    else:
        bin_idx = fn.random.uniform(range=[0, 1], dtype=types.INT32, seed=seed,
                                    shape=tuple() if n == 1 else (n, ))
        bins_to_magnitudes_map = fixed_signed_bin_to_magnitudes(m)
    extra_op_kwargs = extra_op_kwargs or {}
    op_common_kwargs = {
        "num_bins": num_magnitude_bins,
        "extra_op_kwargs": extra_op_kwargs,
        "bins_to_magnitudes_map": bins_to_magnitudes_map,
    }
    op_idx = operation_idx_random_choice(len(ops), n, seed)
    for level_idx in range(n):
        if not use_signed_magnitudes or n == 1:
            level_bin_idx = bin_idx
        else:
            level_bin_idx = bin_idx[level_idx]
        op_kwargs = dict(samples=samples, bin_idx=level_bin_idx, **op_common_kwargs)
        level_op_idx = op_idx if n == 1 else op_idx[level_idx]
        samples = apply_operators_by_idx(ops, level_op_idx, op_kwargs)
    return samples
