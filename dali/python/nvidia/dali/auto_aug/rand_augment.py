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
    "posterize": a.posterize.augmentation((8, 4), False, a.poster_mask_uint8),
    # solarization strength increases with decreasing magnitude (threshold)
    "solarize": a.solarize.augmentation((256, 0)),
    "solarize_add": a.solarize_add.augmentation((0, 110)),
    "invert": a.invert,
    "equalize": a.equalize,
    "auto_contrast": a.auto_contrast,
    "identity": a.identity,
}

non_monotonic_ops = {
    "posterize": a.posterize.augmentation((1, 4), False, a.poster_mask_uint8),
    "solarize": a.solarize.augmentation((0, 256), False, None),
    "brightness": a.brightness.augmentation((0.1, 1.9), False, None),
    "contrast": a.contrast.augmentation((0.1, 1.9), False, None),
    "color": a.color.augmentation((0.1, 1.9), False, None),
    "sharpness": a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
}

rand_augment_suite = ("shear_x", "shear_y", "translate_x", "translate_y", "rotate", "brightness",
                      "contrast", "color", "sharpness", "posterize", "solarize", "solarize_add",
                      "invert", "equalize", "auto_contrast", "identity")


def rand_augment(samples, n, m, num_magnitude_bins=31, shapes=None, fill_value=None,
                 interp_type=None, max_translate_height=250, max_translate_width=250, seed=None,
                 monotonic_mag=True, excluded_ops=None):
    """
    Applies RandAugment (https://arxiv.org/abs/1909.13719) transformations to the provided batch of samples.

    Parameter
    ---------
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout
        and `uint8` type and reside on GPU.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude (strength) of each operation to be applied, it must be an integer
        within `[0, num_magnitude_bins - 1]`.
    shapes: DataNode
        A batch of shapes of the `samples`. If specified, the `translation` operations
        are applied relative to the shape of the sample. Otherwise `max_translate_width`
        and `max_translate_height` constants are used to compute the magnitude of the
        translation.
    fill_value: int, optional
        A value to be used as a padding for images transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    seed: int
        Seed to be used to randomly sample operations (and to negate magnitudes).
    monotonic_mag: bool
        There are two flavours of RandAugment available in different frameworks. For the default
        `monotonic_mag=True` the strengths of operations that accept magnitude increases with
        the increasing magnitudes. If set to False, a different variant is used where some color
        manipulating operations use magnitude ranges that correspond to initial AutoAugment paper.
        There, the `posterize` and `solarize` strength decreases with increasing magnitudes and
        enhance operations (`brightness`, `contrast`, `color`, `sharpness`) use (0.1, 1.9) range,
        which means that the strength decreases the closer the magnitudes are to the center
        of the range. The affected ops are listed in `non_monotonic_ops`.
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
    for name in excluded_ops:
        if name not in rand_augment_suite:
            raise Exception(
                f"The `{name}` was specified in `excluded_ops`, but the rand_augment suite "
                f"does not contain such an augmentation.")
    selected_ops = [ops[name] for name in rand_augment_suite if name not in excluded_ops]
    return apply_rand_augment(selected_ops, samples, n, m, num_magnitude_bins=num_magnitude_bins,
                              seed=seed, extra_op_kwargs=extra_op_kwargs)


def apply_rand_augment(ops, samples, n, m, num_magnitude_bins, seed, extra_op_kwargs=None):
    """
    Applies RandAugment (https://arxiv.org/abs/1909.13719) like transformations but with custom
    set of augmentations.

    Parameter
    ---------
    ops : List[core.Augmentation]
        List of augmentations to be sampled and applied in RandAugment fashion.
    samples : DataNode
        A batch of samples to be processed.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude (strength) of each operation to be applied, it must be an integer
        within `[0, num_magnitude_bins - 1]`.
    seed: int
        Seed to be used to randomly sample operations (and to negate magnitudes).
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
    if num_magnitude_bins <= 1:
        raise Exception(
            f"The number of magnitude bins cannot be less than 1, got {num_magnitude_bins}.")
    if m >= num_magnitude_bins:
        raise Exception(
            f"The magnitude `m` must be an integer within `[0, num_magnitude_bins - 1]` range. "
            f"Got `m={m}`, while the `num_magnitude_bins={num_magnitude_bins}`.")
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
