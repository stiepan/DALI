# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import jax
import jax.dlpack
from jax.sharding import NamedSharding, PositionalSharding, Sharding

from nvidia.dali import ops
from nvidia.dali.ops._operators import python_function


def _with_gpu_dl_tensors_as_arrays(callback):

    def inner(stream, *dl_tensors):
        ts = tuple(jax.dlpack.from_dlpack(t) for t in dl_tensors)
        out = callback(*ts)
        if out is not None:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(jax.dlpack.to_dlpack(t, stream=stream) for t in out)

    return inner


def _with_cpu_dl_tensors_as_arrays(callback):

    def inner(*dl_tensors):
        ts = tuple(jax.dlpack.from_dlpack(t) for t in dl_tensors)
        out = callback(*ts)
        if out is not None:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(jax.dlpack.to_dlpack(t) for t in out)

    return inner


def _with_sharding(callback, sharding: Sharding):
    if jax.local_device_count() != 1:
        raise NotImplementedError(
            f"Currently, the `jax_python_function` supports only global/multiprocessing sharding. The number of local devices seen by the process must be 1, got {jax.local_device_count()}"
        )

    if not isinstance(sharding, (NamedSharding, PositionalSharding)):
        raise ValueError(
            f"The value passed as `sharding` must be an instance of `NamedSharding` or `PositionalSharding`, got value of a type {type(sharding)}"
        )

    def as_sharded_array(array):
        array_shape = array.shape
        if isinstance(sharding, NamedSharding):
            global_shape = (sharding.mesh.size * array_shape[0], *array_shape[1:])
        else:
            global_shape = (sharding.shape[0] * array_shape[0], *array_shape[1:])
        return jax.make_array_from_single_device_arrays(global_shape, sharding, [array])

    def as_single_device_array(sharded_array):
        local_arrays = [x.data for x in sharded_array.addressable_shards]
        if len(local_arrays) != 1:
            raise ValueError(
                f"The function returned multiple shards, expected exactly one, got {len(local_arrays)}."
            )
        return local_arrays[0]

    def inner(*arrays):
        sharded_arrays = tuple(as_sharded_array(array) for array in arrays)
        out = callback(*sharded_arrays)
        if out is not None:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(as_single_device_array(t) for t in out)

    return inner


def _jax_callback_wrapper(function, device, sharding):

    assert device in ("cpu", "gpu")

    dl_pack_wrapper = (
        _with_cpu_dl_tensors_as_arrays if device == "cpu" else _with_gpu_dl_tensors_as_arrays
    )
    return dl_pack_wrapper(function if sharding is None else _with_sharding(function, sharding))


class JaxPythonFunction(
    python_function._get_base_impl("JaxPythonFunction", "JaxPythonFunctionImpl")
):
    ops.register_cpu_op("JaxPythonFunction")
    ops.register_gpu_op("JaxPythonFunction")

    def __init__(self, function, device, sharding: Optional[Sharding] = None, **kwargs):
        self._sharding = sharding
        super().__init__(
            function=_jax_callback_wrapper(function, device, sharding),
            device=device,
            **kwargs,
        )
