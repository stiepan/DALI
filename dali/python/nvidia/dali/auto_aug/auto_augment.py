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

from types import MappingProxyType

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core.utils import operation_idx_random_choice, apply_selected_ops
from nvidia.dali.auto_aug.core.wrapper import ConstMagnitudeAug, ConstSignedMagnitudeAug


class Policy:

    def __init__(self, name, num_bins, augmentations, sub_policies):
        self.name = name
        self.num_bins = num_bins
        # prevent accidental modifications
        self.augmentations = MappingProxyType(augmentations)
        self.sub_policies = tuple(sub_policies)

    def __repr__(self):
        return f"Policy({self.name}, {self.num_bins}, {self.augmentations}, {self.sub_policies})"


auto_augment_image_net_policy = Policy(
    "ImageNet", 11, {
        "shear_x": a.shear_x.augmentation((0, 0.3), True),
        "shear_y": a.shear_y.augmentation((0, 0.3), True),
        "translate_x": a.translate_x.augmentation((0, 0.45), True),
        "translate_y": a.translate_y.augmentation((0, 0.45), True),
        "rotate": a.rotate.augmentation((0, 30), True),
        "brightness": a.brightness.augmentation((0.1, 1.9), False, None),
        "contrast": a.contrast.augmentation((0.1, 1.9), False, None),
        "color": a.color.augmentation((0.1, 1.9), False, None),
        "sharpness": a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
        "posterize": a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
        "solarize": a.solarize.augmentation((0, 256), False),
        "solarize_add": a.solarize_add.augmentation((0, 110), False),
        "invert": a.invert,
        "equalize": a.equalize,
        "auto_contrast": a.auto_contrast,
    }, [
        [("equalize", 0.8, 1), ('shear_y', 0.8, 4)],
        [('color', 0.4, 9), ('equalize', 0.6, 3)],
        [('color', 0.4, 1), ('rotate', 0.6, 8)],
        [('solarize', 0.8, 3), ('equalize', 0.4, 7)],
        [('solarize', 0.4, 2), ('solarize', 0.6, 2)],
        [('color', 0.2, 0), ('equalize', 0.8, 8)],
        [('equalize', 0.4, 8), ('solarize_add', 0.8, 3)],
        [('shear_x', 0.2, 9), ('rotate', 0.6, 8)],
        [('color', 0.6, 1), ('equalize', 1.0, 2)],
        [('invert', 0.4, 9), ('rotate', 0.6, 0)],
        [('equalize', 1.0, 9), ('shear_y', 0.6, 3)],
        [('color', 0.4, 7), ('equalize', 0.6, 0)],
        [('posterize', 0.4, 6), ('auto_contrast', 0.4, 7)],
        [('solarize', 0.6, 8), ('color', 0.6, 9)],
        [('solarize', 0.2, 4), ('rotate', 0.8, 9)],
        [('rotate', 1.0, 7), ('translate_y', 0.8, 9)],
        [('shear_x', 0.0, 0), ('solarize', 0.8, 4)],
        [('shear_y', 0.8, 0), ('color', 0.6, 4)],
        [('color', 1.0, 0), ('rotate', 0.6, 2)],
        [('equalize', 0.8, 4), ('equalize', 0.0, 8)],
        [('equalize', 1.0, 4), ('auto_contrast', 0.6, 2)],
        [('shear_y', 0.4, 7), ('solarize_add', 0.6, 7)],
        [('posterize', 0.8, 2), ('solarize', 0.6, 10)],
        [('solarize', 0.6, 8), ('equalize', 0.6, 1)],
        [('color', 0.8, 6), ('rotate', 0.4, 5)],
    ])


def auto_augment_image_net(samples, shapes=None, fill_value=0, interp_type=None,
                           max_translate_height=250, max_translate_width=250, seed=None):
    augment_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    name = auto_augment_image_net_policy.name
    num_bins = auto_augment_image_net_policy.num_bins
    augments = dict(auto_augment_image_net_policy.augmentations)
    sub_policies = auto_augment_image_net_policy.sub_policies
    if shapes is not None:
        augments["translate_x"] = a.translate_x_no_shape.augmentation((0, max_translate_width))
        augments["translate_y"] = a.translate_y_no_shape.augmentation((0, max_translate_height))
        augment_kwargs["shapes"] = shapes
    policy = Policy(name, num_bins, augments, sub_policies)
    return apply_auto_augment(policy, samples, seed, augment_kwargs)


def apply_auto_augment(policy: Policy, samples, seed=None, augment_kwargs=None):
    if len(policy.sub_policies) == 0:
        return samples
    augmentations = policy.augmentations
    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations.values())
    sub_policies = [[(augmentations[name], p, mag) for name, p, mag in sub_policy]
                    for sub_policy in policy.sub_policies]
    if not use_signed_magnitudes:
        sub_policies = [[(ConstMagnitudeAug(aug, policy.num_bins, mag), p)
                         for aug, p, mag in sub_policy] for sub_policy in sub_policies]
    else:
        bin_idx = ConstSignedMagnitudeAug.get_bins(1, seed)
        sub_policies = [[(ConstSignedMagnitudeAug(aug, policy.num_bins, bin_idx, mag), p)
                         for aug, p, mag in sub_policy] for sub_policy in sub_policies]
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    rand_res = fn.random.uniform(shape=(max_policy_len, ))
    op_kwargs = dict(samples=samples, rand_res=rand_res, **augment_kwargs)
    sub_policies = [apply_sub_policy(sub_policy) for sub_policy in sub_policies]
    policy_id = operation_idx_random_choice(len(sub_policies), 1, seed)
    return apply_selected_ops(sub_policies, policy_id, op_kwargs)


def apply_sub_policy(sub_policy):

    def inner(samples, rand_res, **kwargs):
        for i, (augmentation, p) in enumerate(sub_policy):
            if rand_res[i] <= p:
                samples = augmentation(samples, **kwargs)
        return samples

    return inner