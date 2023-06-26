import warnings

from nvidia.dali.auto_aug.rand_augment import get_rand_augment_suite, get_rand_augment_non_monotonic_suite, \
    _forbid_unused_kwargs, signed_bin, _pretty_select
from nvidia.dali import fn, types
import nvidia.dali.math as dali_math


def apply_rand_augment(augmentations, data, n, m, num_magnitude_bins=31, seed=None, std_dev=None,
                       apply_prob=None, **kwargs):
    if not isinstance(n, int) or n < 0:
        raise Exception(
            f"The number of operations to apply `n` must be a non-negative integer, got {n}.")
    if not isinstance(num_magnitude_bins, int) or num_magnitude_bins < 1:
        raise Exception(
            f"The `num_magnitude_bins` must be a positive integer, got {num_magnitude_bins}.")
    if not isinstance(m, int) or not 0 <= m < num_magnitude_bins:
        raise Exception(f"The magnitude bin `m` must be an integer from "
                        f"`[0, {num_magnitude_bins - 1}]` range. Got {m}.")
    if n == 0:
        warnings.warn(
            "The `apply_rand_augment` was called with `n=0`, "
            "no augmentation will be applied.", Warning)
        return data
    if len(augmentations) == 0:
        raise Exception("The `augmentations` list cannot be empty, unless n=0. "
                        "Got empty list in `apply_rand_augment` call.")
    shape = tuple() if n == 1 else (n, )
    op_idx = fn.random.uniform(values=list(range(len(augmentations))), seed=seed, shape=shape,
                               dtype=types.INT32)
    if std_dev is not None:
        random_dev = fn.random.normal(mean=0, stddev=std_dev, seed=seed, shape=shape,
                                      dtype=types.INT32)
        m = random_dev + m
        m = dali_math.clamp(m, 0, num_magnitude_bins - 1)

    should_apply = fn.random.coin_flip(probability=apply_prob)

    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    mag_bin = signed_bin(m, seed=seed, shape=shape) if use_signed_magnitudes else m
    _forbid_unused_kwargs(augmentations, kwargs, 'apply_rand_augment')

    for level_idx in range(n):
        if apply_prob is not None and should_apply:
            level_mag_bin = mag_bin if not use_signed_magnitudes or n == 1 else mag_bin[level_idx]
            op_kwargs = dict(data=data, magnitude_bin=level_mag_bin,
                             num_magnitude_bins=num_magnitude_bins, **kwargs)
            level_op_idx = op_idx if n == 1 else op_idx[level_idx]
            data = _pretty_select(augmentations, level_op_idx, op_kwargs,
                                  auto_aug_name='apply_rand_augment',
                                  ref_suite_name='get_rand_augment_suite')
    return data


def rand_augment(data, n, m, num_magnitude_bins=31, shape=None, fill_value=128, interp_type=None,
                 max_translate_abs=None, max_translate_rel=None, seed=None, monotonic_mag=True,
                 excluded=None, std_dev=None, apply_prob=None):
    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape
    if monotonic_mag:
        augmentations = get_rand_augment_suite(use_shape, max_translate_abs, max_translate_rel)
    else:
        augmentations = get_rand_augment_non_monotonic_suite(use_shape, max_translate_abs,
                                                             max_translate_rel)
    augmentation_names = set(aug.name for aug in augmentations)
    assert len(augmentation_names) == len(augmentations)
    excluded = excluded or []
    if apply_prob is not None:
        excluded += ["identity"]
    if std_dev is not None:
        # The magnitude bins must be intergers for `@augmentation` instances.
        # So introducing more bins can help increase the resolution of different
        # paramaters when the magnitudes can be randomly distorted
        num_magnitude_bins *= 10
        std_dev *= 10
        m *= 10
    for name in excluded:
        if name not in augmentation_names:
            raise Exception(f"The `{name}` was specified in `excluded`, but the RandAugment suite "
                            f"does not contain augmentation with this name. "
                            f"The augmentations in the suite are: {', '.join(augmentation_names)}.")
    selected_augments = [aug for aug in augmentations if aug.name not in excluded]
    return apply_rand_augment(selected_augments, data, n, m, num_magnitude_bins=num_magnitude_bins,
                              seed=seed, std_dev=std_dev, apply_prob=apply_prob, **aug_kwargs)
