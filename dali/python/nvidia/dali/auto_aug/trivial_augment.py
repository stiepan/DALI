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
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core.utils import operation_idx_random_choice, apply_selected_ops, random_bins_to_signed_magnitudes

trivial_augment_wide_ops = {
    "shear_x": a.shear_x.augmentation((0, 0.99), True),
    "shear_y": a.shear_y.augmentation((0, 0.99), True),
    "translate_x": a.translate_x_no_shape.augmentation((0, 32), True),
    "translate_y": a.translate_y_no_shape.augmentation((0, 32), True),
    "rotate": a.rotate.augmentation((0, 135), True),
    "brightness": a.brightness.augmentation((0.01, 1), True, a.shift_enhance_range),
    "contrast": a.contrast.augmentation((0.01, 1), True, a.shift_enhance_range),
    "color": a.color.augmentation((0.01, 1), True, a.shift_enhance_range),
    "sharpness": a.sharpness.augmentation((0.01, 1), True, a.sharpness_kernel),
    "posterize": a.posterize.augmentation((8, 2), False, a.poster_mask_uint8),
    "solarize": a.solarize.augmentation((256, 0)),
    "equalize": a.equalize,
    "auto_contrast": a.auto_contrast,
    "identity": a.identity,
}

trivial_augment_wide_suite = ("shear_x", "shear_y", "translate_x", "translate_y", "rotate",
                              "brightness", "contrast", "color", "sharpness", "posterize",
                              "solarize", "equalize", "auto_contrast", "identity")


def trivial_augment_wide(samples, num_magnitude_bins=31, fill_value=None, interp_type=None,
                         seed=None, excluded_ops=None):
    """
    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the
    provided batch of samples.

    Parameter
    ---------
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    fill_value: int, optional
        A value to be used as a padding for images transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    excluded_ops: List[str], optional
        A list of names of the operations to be excluded from the `rand_augment_suite`.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tuned of the existing ones, you can use the `apply_rand_augment`
        directly, which accepts a list of augmentations.
    extra_op_kwargs:
        A dictionary of extra parameters (for example DataNodes) to be passed to the
        augmentations specified through `ops`. The signature of the augmentations are
        checked for any extra arguments and if the name of the argument matches one from the
        `extra_op_kwargs`, the value is passed as an argument.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    ops = dict(**trivial_augment_wide_ops)
    extra_op_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    excluded_ops = excluded_ops or tuple()
    for name in excluded_ops:
        if name not in trivial_augment_wide_suite:
            raise Exception(
                f"The `{name}` was specified in `excluded_ops`, but the trivial_augment_wide "
                f"suite does not contain such an augmentation.")
    selected_ops = [ops[name] for name in trivial_augment_wide_suite if name not in excluded_ops]
    return apply_trivial_augment(selected_ops, samples, num_magnitude_bins=num_magnitude_bins,
                                 seed=seed, extra_op_kwargs=extra_op_kwargs)


def apply_trivial_augment(ops, samples, num_magnitude_bins, seed, extra_op_kwargs=None):
    """
    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the
    provided batch of samples but with a custom set of augmentations.

    Parameter
    ---------
    ops : List[core.Augmentation]
        List of augmentations to be sampled and applied in TrivialAugment fashion.
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    excluded_ops: List[str], optional
        A list of names of the operations to be excluded from the `rand_augment_suite`.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tuned of the existing ones, you can use the `apply_rand_augment`
        directly, which accepts a list of augmentations.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if num_magnitude_bins <= 1:
        raise Exception(
            f"The number of magnitude bins cannot be less than 1, got {num_magnitude_bins}.")
    if len(ops) == 0:
        return samples
    use_signed_magnitudes = any(op.randomly_negate for op in ops)
    if not use_signed_magnitudes:
        bin_idx = fn.random.uniform(range=[0, num_magnitude_bins - 1], dtype=types.INT32, seed=seed)
        bins_to_magnitudes_map = None
    else:
        num_rand_bins = 2 * num_magnitude_bins
        bin_idx = fn.random.uniform(range=[0, num_rand_bins - 1], dtype=types.INT32, seed=seed)
        bins_to_magnitudes_map = random_bins_to_signed_magnitudes
    op_common_kwargs = {
        "num_bins": num_magnitude_bins,
        "extra_op_kwargs": extra_op_kwargs or {},
        "bins_to_magnitudes_map": bins_to_magnitudes_map,
    }
    op_idx = operation_idx_random_choice(len(ops), 1, seed)
    op_kwargs = dict(samples=samples, bin_idx=bin_idx, **op_common_kwargs)
    return apply_selected_ops(ops, op_idx, op_kwargs)
