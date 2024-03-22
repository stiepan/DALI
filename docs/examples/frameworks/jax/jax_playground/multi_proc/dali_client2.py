import argparse
import logging as log
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from functools import partial
from jax.experimental.shard_map import shard_map


from nvidia.dali import fn, types
from nvidia.dali.plugin.jax import data_iterator


def my_iter_fn(shard_id, num_shards):
    disjoint_for_some_reason = [
        "/home/ktokarski/imgnet_two_classes/ptaszke",
        "/home/ktokarski/imgnet_two_classes/robakzwody",
    ]
    assert num_shards == len(disjoint_for_some_reason)
    encoded, label = fn.readers.file(
        file_root=disjoint_for_some_reason[shard_id],
        random_shuffle=True,
    )
    local_num_classes = 2
    label += shard_id * local_num_classes
    image = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
    image = fn.resize(image, size=(224, 224), interp_type=types.INTERP_LINEAR, antialias=True)
    # image, label = fn.dl_tensor_python_function(
    #     image,
    #     label.gpu(),
    #     function=create_jax_permute_cb(jax_sharding),
    #     batch_processing=True,
    #     synchronize_stream=False,
    #     dl_tensor_stream_aware=True,
    #     output_layouts=["HWC", ""],
    #     num_outputs=2,
    # )
    return image, label


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()

    process_id = args.id
    cluster_size = args.size

    jax.distributed.initialize(
        coordinator_address="localhost:12321", num_processes=cluster_size, process_id=process_id
    )

    log.basicConfig(format=f"PID {process_id}: %(message)s", level=log.INFO)
    mesh = mesh_utils.create_device_mesh(jax.device_count(), jax.devices())
    mesh = Mesh(mesh, axis_names=("device"))
    sharding = NamedSharding(mesh, PartitionSpec("device"))
    print(sharding.mesh.size)

    log.info(f"Sharding: {sharding}")

    my_iter = data_iterator(output_map=["image", "label"], sharding=sharding)(my_iter_fn)
    mi = my_iter(batch_size=16)
    print(mi._pipes)
    print([p.max_batch_size for p in mi._pipes])
    for i, data in enumerate(mi):
        if i >= 2:
            break
        print(data['label'].shape)
        with jax.spmd_mode("allow_all"):
            jax.debug.visualize_array_sharding(data['image'].ravel())



main()