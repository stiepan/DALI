import argparse
import logging as log

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.sharding import PositionalSharding

from functools import partial
from jax.experimental.shard_map import shard_map
from pytools import P


def print_devices_details(devices_list, process_id):
    for device in devices_list:
        log.info(
            f"Id = {device.id}, platform = {device.platform}, "
            f"process_id = {device.process_index}, kind = {device.device_kind}"
        )


def print_devices(process_id):
    log.info(
        f"Local devices = {jax.local_device_count()}, " f"global devices = {jax.device_count()}"
    )

    log.info("All devices: ")
    print_devices_details(jax.devices(), process_id)

    log.info("Local devices:")
    print_devices_details(jax.local_devices(), process_id)


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

    print_devices(process_id=process_id)

    mesh = mesh_utils.create_device_mesh(jax.device_count(), jax.devices())
    # sharding = PositionalSharding(mesh)

    mesh = Mesh(mesh, axis_names=("device"))
    sharding = NamedSharding(mesh, PartitionSpec("device"))

    log.info(f"Sharding: {sharding}")

    dali_local_shards = []
    for id, device in enumerate(jax.local_devices()):
        current_shard = jax.device_put(
            jnp.array([[process_id, 10 * process_id, 100 * process_id]], dtype=jnp.int32),
            jax.local_devices()[0],
        )

        assert current_shard.devices() == {device}, f"{current_shard.devices()}, {[device]}"

        dali_local_shards.append(current_shard)

    assert len(dali_local_shards) == 1

    dali_sharded_array = jax.make_array_from_single_device_arrays(
        shape=(jax.device_count(), 3), sharding=sharding, arrays=dali_local_shards
    )

    with jax.spmd_mode("allow_all"):
        jax.debug.visualize_array_sharding(dali_sharded_array.ravel())

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(PartitionSpec("device"),),
        out_specs=PartitionSpec("device"),
    )
    def compute_sth(process_id_array):
        a4x4 = jnp.full((4, 3), 1) * process_id_array
        # return a4x4
        # return jax.lax.all_gather(a4x4, "device")
        return jax.lax.psum(a4x4, "device")

    out = compute_sth(dali_sharded_array)
    device_buffers = [x.data for x in out.addressable_shards]
    print(device_buffers)

    # out = jax.pmap(compute_sth)(dali_sharded_array)
    # print(out)

    # device_buffers = [x.data for x in dali_sharded_array.addressable_shards]
    # assert len(device_buffers) == jax.local_device_count()

    # for id, buffer in enumerate(device_buffers):
    #     assert buffer == jnp.array([[process_id, 10 * process_id, 100 * process_id]],)
    #     assert buffer.devices() == {jax.local_devices()[id]}


main()
