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


class Policy:

    def __init__(self, name, num_bins, augmentations, sub_policies):
        self.name = name
        self.num_bins = num_bins
        self.augmentations = augmentations
        self.sub_policies = sub_policies

    def __repr__(self):
        return f"Policy({self.name}, {self.num_bins}, {self.augmentations}, {self.sub_policies})"


auto_augment_image_net_policy = Policy(
    "ImageNet", 10, {
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


def apply_auto_augment(policy, samples, shapes=None, fill_value=None, interp_type=None, seed=None):
    pass


def run_sub_policy():
    pass