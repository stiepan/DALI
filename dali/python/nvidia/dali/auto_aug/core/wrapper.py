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


class Augmentation:

    def __init__(self, op, mag_range=None, as_param=None, randomly_negate=None, param_device=None):
        self._op = op
        self._mag_range = mag_range
        self._as_param = as_param
        self._randomly_negate = randomly_negate
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

    def augmentation(self, mag_range=None, as_param=None, randomly_negate=None, param_device=None,
                     augmentation_cls=None):
        cls = augmentation_cls or self.__class__
        config = {name: value for name, value in self._get_config()}
        for name, value in (
            ('mag_range', mag_range),
            ('as_param', as_param),
            ('randomly_negate', randomly_negate),
            ('param_device', param_device),
        ):
            if value is not None:
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

    def __call__(self, samples, bin_idx, num_bins, mag_to_param_range, extra_op_kwargs=None):
        if not isinstance(bin_idx, _DataNode):
            params = self._get_fixed_magnitude(bin_idx, num_bins)
        else:
            params = self._get_magnitude_data_node(bin_idx, num_bins, mag_to_param_range)
        return self._call_op(samples, params, extra_op_kwargs)

    def _get_mag_range(self, num_bins):
        mag_range = self.mag_range
        if isinstance(mag_range, tuple) and len(mag_range) == 2:
            lo, hi = mag_range
            return np.linspace(lo, hi, num_bins, dtype=np.float32)
        if len(mag_range) != num_bins:
            raise Exception(
                f"Got `mag_range` of length {len(mag_range)} while the `num_bins` specified is {num_bins}."
            )
        return mag_range

    def _get_fixed_magnitude(self, bin_idx, num_bins):
        assert not isinstance(bin_idx, _DataNode)
        magnitudes = self._get_mag_range(num_bins)
        param = np.array(self.as_param(magnitudes[bin_idx]))
        return types.Constant(param, device=self.param_device)

    def _get_magnitude_data_node(self, bin_idx, num_bins, mag_to_param_range):
        magnitudes = self._get_mag_range(num_bins)
        as_param = self.as_param
        params = mag_to_param_range(magnitudes, as_param, self.randomly_negate)
        params = types.Constant(params, device=self.param_device)
        return params[bin_idx]

    def _call_op(self, samples, params, extra_op_kwargs):
        extra_op_kwargs = extra_op_kwargs or {}
        fun_args = inspect.getfullargspec(self._op).args
        kwargs = {name: param for name, param in extra_op_kwargs.items() if name in fun_args}
        return self._op(samples, params, **kwargs)


def augmentation(function=None, mag_range=None, as_param=None, randomly_negate=None, param_device=None,
                 augmentation_cls=None):

    def decorator(function):
        cls = augmentation_cls or Augmentation
        return cls(function, mag_range=mag_range, as_param=as_param, randomly_negate=randomly_negate,
                   param_device=param_device)

    if function is None:
        return decorator
    else:
        return decorator(function)
