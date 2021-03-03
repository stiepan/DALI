# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# it is enough to just import all functions from test_internals_operator_external_source
# nose will query for the methods available and will run them
# the test_internals_operator_external_source is 99% the same for cupy and numpy tests
# so it is better to store everything in one file and just call `use_cupy` to switch between the default numpy and cupy

import torch
from nose.tools import raises, with_setup

from test_pool_utils import *
from test_external_source_parallel_utils import *


class ExtCallbackTorch(ExtCallback):
    def __call__(self, sample_info):
        return torch.tensor(super().__call__(sample_info))


@raises(RuntimeError)
@with_setup(setup_function, teardown_function)
def test_pytorch_cuda_context():
    # Create a dummy torch CUDA tensor so we acquire CUDA context
    cuda0 = torch.device('cuda:0')
    _ = torch.ones([1, 1], dtype=torch.float32, device=cuda0)
    callback = ExtCallback((4, 5), 10, np.int32)
    pipe = create_pipe(callback, 'cpu', 5, py_num_workers=6,
                       py_start_method='fork', parallel=True)
    pipe.start_py_workers()
    capture_processes(pipe._py_pool)


@with_setup(setup_function, teardown_function)
def test_pytorch():
    yield from check_spawn_with_callback(ExtCallbackTorch)


class ExtCallbackTorchCuda(ExtCallback):
    def __call__(self, sample_info):
        return torch.tensor(super().__call__(sample_info), device=torch.device('cuda:0'))


@raises(Exception)
@with_setup(setup_function, teardown_function)
def test_pytorch_cuda():
    callback = ExtCallbackTorchCuda((4, 5), 10, np.int32)
    pipe = create_pipe(callback, 'cpu', 5, py_num_workers=6,
                       py_start_method='spawn', parallel=True)
    build_and_run_pipeline(pipe)
