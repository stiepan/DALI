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

import inspect

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.data_node import DataNode as _DataNode

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


def _np_wrap(mag):
    return np.array(mag)


class _DummyParam:
    """Use DummyParam as a kwarg default when it matters to distinguish between `kwarg=None`
    and not specifying the kwarg"""


class Augmentation:

    def __init__(self, op, mag_range=None, randomly_negate=None, as_param=None, param_device=None):
        self._op = op
        self._mag_range = mag_range
        self._randomly_negate = randomly_negate
        self._as_param = as_param
        self._param_device = param_device

    @property
    def mag_range(self):
        return self._mag_range or (0, 0)

    @property
    def as_param(self):
        return self._as_param or _np_wrap

    @property
    def randomly_negate(self):
        return self._randomly_negate or False

    @property
    def param_device(self):
        return self._param_device or "cpu"

    def augmentation(self, mag_range=_DummyParam, randomly_negate=_DummyParam, as_param=_DummyParam,
                     param_device=_DummyParam, augmentation_cls=None):
        """
        The method to override augmentation parameters specified with `@augmentation` decorator.
        Returns a new augmentation with the original operation decorated but updated parameters.
        Parameters that are not specified are inherited from the initial augmentation.
        """
        cls = augmentation_cls or self.__class__
        config = {name: value for name, value in self._get_config()}
        for name, value in (
            ('mag_range', mag_range),
            ('as_param', as_param),
            ('randomly_negate', randomly_negate),
            ('param_device', param_device),
        ):
            if value is not _DummyParam:
                config[name] = value
        return cls(self._op, **config)

    def _get_config(self):
        return [(name, value) for name, value in (
            ('mag_range', self._mag_range),
            ('as_param', self._as_param),
            ('randomly_negate', self._randomly_negate),
            ('param_device', self._param_device),
        )]

    def __repr__(self):
        aug_params_repr = [repr(self._op)]
        config = [(name, val) for name, val in self._get_config() if val is not None]
        config_reprs = [f"{name}={repr(param)}" for name, param in config]
        aug_params_repr.extend(config_reprs)
        return f"Augmentation({', '.join(aug_params_repr)})"

    def __call__(self, samples, params, **kwargs):
        fun_args = inspect.getfullargspec(self._op).args[2:]
        op_kwargs = {name: param for name, param in kwargs.items() if name in fun_args}
        return self._op(samples, params, **op_kwargs)

    def get_mag_range(self, num_bins):
        mag_range = self.mag_range
        if isinstance(mag_range, tuple) and len(mag_range) == 2:
            lo, hi = mag_range
            return np.linspace(lo, hi, num_bins, dtype=np.float32)
        if len(mag_range) != num_bins:
            raise Exception(
                f"Got `mag_range` of length {len(mag_range)} while the `num_bins` specified is {num_bins}."
            )
        return mag_range


def augmentation(function=None, *, mag_range=None, randomly_negate=None, as_param=None,
                 param_device=None, augmentation_cls=None):
    """
    A decorator turning DALI operation into an augmentation that can be used with the
    `auto_aug` transformations such as RandAugment.

    Parameter
    ---------
    mag_range : (int, int)
        Specifies the range of applicable magnitudes for the operation.
    randomly_negate: bool
        If true, the magnitude from the mag_range will be randomly negated for every sample.
    as_param: callable
        A callback that transforms the magnitude into a parameter that will be passed to the decorated
        operation instead of the plain magnitude. This, the parameters for the range of magnitudes
        can be computed once in advance and stored as a Constant node.
    param_device: str
        A "cpu" or "gpu", describes where to store the precomputed parameters.

    Returns
    -------
    Augmentation
        The operation wrapped with the Augmentation class so that it can be used with the `auto_aug`
        transforms.
    """

    def decorator(function):
        cls = augmentation_cls or Augmentation
        return cls(function, mag_range=mag_range, as_param=as_param,
                   randomly_negate=randomly_negate, param_device=param_device)

    if function is None:
        return decorator
    else:
        if not callable(function):
            raise Exception(f"The `@augmentation` decorator was used to decorate the object that "
                            f"is not callable: {function}.")
        return decorator(function)


class ParametrizedAugmentation:
    """
    Wraps augmentation instance and concrete magnitude to automatically provide
    the augmentation with parameter when called with samples.
    """

    def __init__(self, augmentation: Augmentation, num_magnitude_bins: int, bin_idx):
        if num_magnitude_bins < 1:
            raise Exception(f"The `num_magnitude_bins` must be a positive integer. "
                            f"Got {num_magnitude_bins}.")
        if not isinstance(bin_idx, _DataNode) and not 0 <= bin_idx < num_magnitude_bins:
            raise Exception(f"Expected the magnitude bin from range `[0, num_magnitude_bins - 1]`."
                            f"Got the magnitude bin index {bin_idx}, but the "
                            f"`num_magnitude_bins` is {num_magnitude_bins}.")
        self.augmentation = augmentation
        self.num_magnitude_bins = num_magnitude_bins
        self.bin_idx = bin_idx

    def __repr__(self):
        return f"{self.__class__.__name__}({self.augmentation}, {self.num_magnitude_bins}, {self.bin_idx})"

    def __call__(self, samples, **kwargs):
        param = self.get_param()
        return self.augmentation(samples, param, **kwargs)

    def get_param(self):
        raise NotImplementedError


class ConstMagnitudeAug(ParametrizedAugmentation):

    def get_param(self):
        as_param = self.augmentation.as_param
        magnitudes = self.augmentation.get_mag_range(self.num_magnitude_bins)
        magnitude = magnitudes[self.bin_idx]
        param = np.array(as_param(magnitude))
        return types.Constant(param, device=self.augmentation.param_device)


class ConstSignedMagnitudeAug(ParametrizedAugmentation):

    @classmethod
    def get_bins(cls, num_levels: int, seed: int):
        return fn.random.uniform(range=[0, 1], dtype=types.INT32, seed=seed,
                                 shape=tuple() if num_levels == 1 else (num_levels, ))

    def __init__(self, augmentation: Augmentation, num_magnitude_bins: int, bin_idx: _DataNode,
                 fixed_magnitude_bin: int):
        super().__init__(augmentation, num_magnitude_bins, bin_idx)
        self.fixed_magnitude_bin = fixed_magnitude_bin
        if not 0 <= fixed_magnitude_bin < self.num_magnitude_bins:
            raise Exception(f"Expected the magnitude bin from range `[0, num_magnitude_bins - 1]`."
                            f"Got the magnitude bin index {fixed_magnitude_bin}, but the "
                            f"`num_magnitude_bins` is {self.num_magnitude_bins}.")

    def get_param(self):
        as_param = self.augmentation.as_param
        magnitudes = self.augmentation.get_mag_range(self.num_magnitude_bins)
        magnitudes = [magnitudes[self.fixed_magnitude_bin]]
        magnitudes = remap_bins_to_signed_magnitudes(magnitudes, self.augmentation.randomly_negate)
        params = np.array([as_param(magnitude) for magnitude in magnitudes])
        params = types.Constant(params, device=self.augmentation.param_device)
        return params[self.bin_idx]


class RandomMagnitudeAug(ParametrizedAugmentation):

    @classmethod
    def get_bins(cls, num_magnitude_bins: int, num_levels: int, seed: int):
        return fn.random.uniform(range=[0, num_magnitude_bins - 1], dtype=types.INT32, seed=seed,
                                 shape=tuple() if num_levels == 1 else (num_levels, ))

    def get_param(self):
        as_param = self.augmentation.as_param
        magnitudes = self.augmentation.get_mag_range(self.num_magnitude_bins)
        params = np.array([as_param(magnitude) for magnitude in magnitudes])
        params = types.Constant(params, device=self.augmentation.param_device)
        return params[self.bin_idx]


class RandomSignedMagnitudeAug(ParametrizedAugmentation):

    @classmethod
    def get_bins(cls, num_magnitude_bins: int, num_levels: int, seed: int):
        num_bins = 2 * num_magnitude_bins  # encode the sign in parity
        return fn.random.uniform(range=[0, num_bins - 1], dtype=types.INT32, seed=seed,
                                 shape=tuple() if num_levels == 1 else (num_levels, ))

    def get_param(self):
        as_param = self.augmentation.as_param
        magnitudes = self.augmentation.get_mag_range(self.num_magnitude_bins)
        magnitudes = remap_bins_to_signed_magnitudes(magnitudes, self.augmentation.randomly_negate)
        params = np.array([as_param(magnitude) for magnitude in magnitudes])
        params = types.Constant(params, device=self.augmentation.param_device)
        return params[self.bin_idx]


def remap_bins_to_signed_magnitudes(magnitudes, randomly_negate):

    def remap_bin_idx(bin_idx):
        magnitude = magnitudes[bin_idx // 2]
        if randomly_negate and bin_idx % 2:
            magnitude = -magnitude
        return magnitude

    return np.array([remap_bin_idx(bin_idx) for bin_idx in range(2 * len(magnitudes))])
