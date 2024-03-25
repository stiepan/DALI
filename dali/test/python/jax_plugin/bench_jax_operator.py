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

import argparse
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
from nvidia.dali import pipeline_def, fn, types
import nvidia.dali.plugin.jax as dax

from jax_op_utils import jax_color_twist


dali_extra_path = os.environ["DALI_EXTRA_PATH"]
images_dir = os.path.join(dali_extra_path, "db", "single", "jpeg")

# loading part - file, external_source no-copy, constant
# processing part - color_twist, cmn, color_twist + cmn

types_map = {
    np.uint8: types.DALIDataType.UINT8,
    np.int16: types.DALIDataType.INT16,
    np.float32: types.DALIDataType.FLOAT,
}

def loader_file(batch_size, device, dtype):
    img, _ = fn.readers.file(name="Reader", file_root=images_dir, random_shuffle=True, seed=42)
    img = fn.decoders.image(img, device="cpu" if device == "cpu" else "mixed")
    img = fn.resize(img, size=(224, 224))
    if dtype != np.uint8:
        img = fn.cast(img, dtype=types_map[dtype])
        if dtype == np.float32:
            img = img / 255
    return img


def loader_constant(batch_size, device, dtype):
    rng = np.random.default_rng(42)
    data = dtype(rng.uniform(size=(224, 224, 3)) * (1 if dtype == np.float32 else 255))
    return types.Constant(data, dtype=types_map[dtype], device=device, layout="HWC")


def loader_external_source(batch_size, device, dtype):

    rng = np.random.default_rng(42)
    batch = [dtype(rng.uniform(size=(224, 224, 3)) * (1 if dtype == np.float32 else 255)) for _ in range(batch_size)]
    return fn.external_source(lambda: batch, device=device, layout="HWC", dtype=types_map[dtype])


def get_loader(name, batch_size, device, dtype):

    def inner():
        if name == "file":
            return loader_file(batch_size, device, dtype)
        elif name == "constant":
            return loader_constant(batch_size, device, dtype)
        elif name == "external_source":
            return loader_external_source(batch_size, device, dtype)
        else:
            raise ValueError(f"Unknown loader: {name}")

    return inner


@pipeline_def(device_id=0, num_threads=4, seed=42)
def pipeline(loader, device, use_jax):
    img = loader()
    bcs = fn.random.uniform(range=[0.1, 1.9], shape=3, seed=42)
    hue = fn.random.uniform(range=[0, 180], seed=42)
    if use_jax:
        if device == "gpu":
            bcs, hue = bcs.gpu(), hue.gpu()
        img = dax.fn.jax_python_function(
            img, bcs, hue, function=jax.jit(jax.vmap(jax_color_twist)), output_layouts="HWC")
    else:
        img = fn.color_twist(img, brightness=bcs[0], contrast=bcs[1], saturation=bcs[2], hue=hue)
    return img


def run_pipeline(num_iters, p):
    p.build()
    first_iter_start = time.time()
    p.run()
    start = first_iter_end = time.time()
    for _ in range(num_iters):
        p.run()
    end = time.time()
    return first_iter_end - first_iter_start, end - start


def run_case(num_iters, loader_name, device, dtype, batch_size):
    import gc
    loader = get_loader(loader_name, batch_size, device, dtype)
    dali_first, dali_run = run_pipeline(num_iters, pipeline(loader, device, use_jax=False, batch_size=batch_size))
    dali_jax_first, dali_jax_run = run_pipeline(num_iters, pipeline(loader, device, use_jax=True, batch_size=batch_size))
    gc.collect()
    factor = (dali_jax_run / dali_run)
    #TODO(ktokarski) Add a couple of iterations and compare the results to assure correctness

    print(
        f"Run {num_iters} iterations on {device}, dtype {dtype}, loader {loader_name}, batch_size {batch_size}: \n"
        f"DALI: {dali_run:.2f}s, JAX: {dali_jax_run:.2f}s.\n"
        f"JAX/DALI run: {factor:.2f}.\n"
        f"First iteration: DALI: {dali_first:.2f}s, JAX: {dali_jax_first:.2f}s.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=100, required=False)
    parser.add_argument("--device", type=str, default="gpu", required=False)
    args = parser.parse_args()
    num_iters = args.num_iters
    device = args.device
    assert device in ("cpu", "gpu")

    for dtype in (np.uint8, np.int16, np.float32):
        # 1. file reader is a baseline
        # 2. constant should remove any loading from the picture but will suffer
        #    from the need to copy the batch into a contiguous batch
        # 3. external source may reveal cost of the python op and the external source
        #    competing for the gil
        for loader_name in ("file", "constant", "external_source"):
            #TODO(ktokarski) Add cmn and cmn + color_twist
            for batch_size in (1, 32, 128):
                run_case(num_iters, loader_name, device, dtype, batch_size)



main()