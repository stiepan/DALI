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
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core.utils import operation_idx_random_choice, apply_operators_by_idx, fixed_signed_bin_to_magnitudes

rand_augment_ops = {
    "shear_x": a.shear_x.augmentation((0, 0.3), True),
    "shear_y": a.shear_y.augmentation((0, 0.3), True),
    "translate_x": a.translate_x.augmentation((0, 0.45), True),
    "translate_y": a.translate_y.augmentation((0, 0.45), True),
    "rotate": a.rotate.augmentation((0, 30), True),
    "brightness": a.brightness.augmentation((0, 0.9), True, a.shift_enhance_range),
    "contrast": a.contrast.augmentation((0, 0.9), True, a.shift_enhance_range),
    "color": a.color.augmentation((0, 0.9), True, a.shift_enhance_range),
    "sharpness": a.sharpness.augmentation((0, 0.9), True, a.sharpness_kernel),
    "posterize": a.posterize.augmentation((0, 7), False, a.poster_mask_uint8),
    # solarization strength increases with decreasing magnitude (threshold)
    "solarize": a.solarize.augmentation((256, 0)),
    "solarize_add": a.solarize_add.augmentation((0, 110)),
    "invert": a.invert,
    "equalize": a.equalize,
    "auto_contrast": a.auto_contrast,
    "identity": a.identity,
}

# There are two flavours of RandAugment available in different frameworks, one that
# makes sure that strength of each operation corresponds to increasing magnitudes
# and one that uses magnitudes similar to the ones from AutoAugment. In the latter variant,
# the posterize and solarize strength decreases, while the "enhance" operators strength
# decreases the closer the magnitude is to the center of the range.
non_monotonic_ops = {
    "posterize": a.posterize.augmentation((7, 0), False, as_param=a.poster_mask_uint8),
    "solarize": a.solarize.augmentation((0, 256), False, as_param=None),
    "brightness": a.brightness.augmentation((0.1, 1.9), False, as_param=None),
    "contrast": a.contrast.augmentation((0.1, 1.9), False, as_param=None),
    "color": a.color.augmentation((0.1, 1.9), False, as_param=None),
    "sharpness": a.sharpness.augmentation((0.1, 1.9), False, as_param=a.sharpness_kernel_shifted),
}

rand_augment_suite = ("shear_x", "shear_y", "translate_x", "translate_y", "rotate", "brightness",
                      "contrast", "color", "sharpness", "posterize", "solarize", "solarize_add",
                      "invert", "equalize", "auto_contrast", "identity")


def rand_augment(samples, n, m, num_magnitude_bins=31, shapes=None, fill_value=None,
                 interp_type=None, max_translate_width=250, max_translate_height=250, seed=None,
                 monotonic_mag=True, excluded_ops=None):
    ops = dict(**rand_augment_ops)
    extra_op_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    if shapes is not None:
        if not isinstance(shapes, _DataNode):
            raise Exception(
                f"The `shapes` parameter must be an output of DALI operator (DataNode) that "
                f"describes height and width of the samples node of DALI graph , got {shapes}.")
        extra_op_kwargs["shapes"] = shapes
    else:
        ops["translate_x"] = a.translate_x_no_shape.augmentation((0, max_translate_width))
        ops["translate_y"] = a.translate_y_no_shape.augmentation((0, max_translate_height))
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
            f"Got `m={m}`, while the `num_magnitude_bins={num_magnitude_bins}`")
    if len(ops) == 0:
        return samples
    use_signed_magnitudes = any(op.randomly_negate for op in ops)
    if not use_signed_magnitudes:
        bin_idx, bins_to_magnitudes_map = m, None
    else:
        bin_idx = fn.random.uniform(range=[0, 1], dtype=types.INT32, seed=seed,
                                    shape=tuple() if n == 1 else (n, ))
        bins_to_magnitudes_map = fixed_signed_bin_to_magnitudes(m)
    op_common_kwargs = {
        "num_bins": num_magnitude_bins,
        "extra_op_kwargs": extra_op_kwargs or {},
        "bins_to_magnitudes_map": bins_to_magnitudes_map,
    }
    op_idx = operation_idx_random_choice(len(ops), n, seed)
    for level_idx in range(n):
        level_bin_idx = bin_idx if not use_signed_magnitudes or n == 1 else bin_idx[level_idx]
        op_kwargs = dict(samples=samples, bin_idx=level_bin_idx, **op_common_kwargs)
        level_op_idx = op_idx if n == 1 else op_idx[level_idx]
        samples = apply_operators_by_idx(ops, level_op_idx, op_kwargs)
    return samples
