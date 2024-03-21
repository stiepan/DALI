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


from nvidia.dali import fn, types, pipeline_def


@jax.vmap
def jax_permute_sample(key, image, label):
    should_flip = jax.random.choice(key, 2)
    return jax.lax.cond(should_flip, lambda: (jax.lax.pshuffle(image, "device", [1, 0]), jax.lax.pshuffle(label, "device", [1, 0])), lambda: (image, label))


def jax_permute_batch(key, images, labels):
    """The image is a 4D tensor of shape (batch_size, height, width, channels), the returned tensor is a 4D tensor of shape (batch_size, height, width, channels) where elements along batch dimension are randomly permuted"""
    # key = jax.random.PRNGKey(0)
    # print(labels.shape)
    # perm = jax.random.permutation(key, 2)
    keys = jax.random.split(key, images.shape[0])
    images, labels = jax_permute_sample(keys, images, labels)
    # images = jax.lax.pshuffle(images, "device", perm)
    # labels = jax.lax.pshuffle(labels, "device", perm)
    # labels = jax.lax.ppermute(labels, "device", jnp.array([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]))
    # images = jax.random.shuffle(key, images)
    # labels = jax.random.shuffle(key, labels)
    # all_labels = jax.lax.all_gather(labels, "device", tiled=True)
    # print(all_labels.shape)
    return images, labels


def create_jax_permute_cb(sharding, seed=42):

    @jax.jit
    @partial(
        shard_map,
        mesh=sharding.mesh,
        in_specs=(PartitionSpec(None), PartitionSpec("device"), PartitionSpec("device"),),
        out_specs=(PartitionSpec("device"), PartitionSpec("device")),
    )
    def actual_cb(key, images, labels):
        return jax_permute_batch(key, images, labels)

    key = jax.random.PRNGKey(seed)

    def cb(images, labels):
        nonlocal key
        key, sub_key = jax.random.split(key)
        images_shape = images.shape
        images_sharded = jax.make_array_from_single_device_arrays(
            shape=((jax.device_count() * images_shape[0],) + images_shape[1:]),
            sharding=sharding,
            arrays=[images]
        )
        # device_buffers = [x.data for x in images_sharded.addressable_shards]
        # buffer_devices = [x.devices() for x in device_buffers]
        # log.info(f"{device_buffers}, {buffer_devices}")
        labels_shape = labels.shape
        labels_sharded = jax.make_array_from_single_device_arrays(
            shape=((jax.device_count() * labels_shape[0],) + labels_shape[1:]),
            sharding=sharding,
            arrays=[labels]
        )
        # device_buffers = [x.data for x in labels_sharded.addressable_shards]
        # buffer_devices = [x.devices() for x in device_buffers]
        # log.info(f"{device_buffers}, {buffer_devices}")
        images_sharded, labels_sharded = actual_cb(sub_key, images_sharded, labels_sharded)

        local_images = [x.data for x in images_sharded.addressable_shards]
        assert len(local_images) == 1
        local_labels = [x.data for x in labels_sharded.addressable_shards]
        assert len(local_labels) == 1
        return local_images[0], local_labels[0]


    return cb


@pipeline_def(seed=42, num_threads=4)
def pipeline(shard_id, num_shards, jax_sharding):
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
    image, label = fn.dl_tensor_python_function(
        image,
        label.gpu(),
        function=create_jax_permute_cb(jax_sharding),
        batch_processing=True,
        synchronize_stream=False,
        dl_tensor_stream_aware=True,
        output_layouts=["HWC", ""],
        num_outputs=2,
    )
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

    print_devices(process_id=process_id)

    mesh = mesh_utils.create_device_mesh(jax.device_count(), jax.devices())
    # sharding = PositionalSharding(mesh)

    mesh = Mesh(mesh, axis_names=("device"))
    sharding = NamedSharding(mesh, PartitionSpec("device"))

    log.info(f"Sharding: {sharding}")

    p = pipeline(batch_size=8, shard_id=process_id, num_shards=cluster_size, device_id=0, jax_sharding=sharding)
    p.build()
    for _ in range(10):
        out, labels = p.run()
        labels = np.array([s for s in labels.as_cpu()]).squeeze()
        log.info(f"labels: {labels}")

    # dali_local_shards = []
    # for id, device in enumerate(jax.local_devices()):
    #     current_shard = jax.device_put(
    #         jnp.array([[process_id, 10 * process_id, 100 * process_id]], dtype=jnp.int32),
    #         jax.local_devices()[0],
    #     )

    #     assert current_shard.devices() == {device}, f"{current_shard.devices()}, {[device]}"

    #     dali_local_shards.append(current_shard)

    # assert len(dali_local_shards) == 1

    # dali_sharded_array = jax.make_array_from_single_device_arrays(
    #     shape=(jax.device_count(), 3), sharding=sharding, arrays=dali_local_shards
    # )

    # with jax.spmd_mode("allow_all"):
    #     jax.debug.visualize_array_sharding(dali_sharded_array.ravel())

    # @partial(
    #     shard_map,
    #     mesh=mesh,
    #     in_specs=(PartitionSpec("device"),),
    #     out_specs=PartitionSpec("device"),
    # )
    # def compute_sth(process_id_array):
    #     a4x4 = jnp.full((4, 3), 1) * process_id_array
    #     # return a4x4
    #     # return jax.lax.all_gather(a4x4, "device")
    #     return jax.lax.psum(a4x4, "device")

    # out = compute_sth(dali_sharded_array)
    # device_buffers = [x.data for x in out.addressable_shards]
    # print(device_buffers)

    # out = jax.pmap(compute_sth)(dali_sharded_array)
    # print(out)

    # device_buffers = [x.data for x in dali_sharded_array.addressable_shards]
    # assert len(device_buffers) == jax.local_device_count()

    # for id, buffer in enumerate(device_buffers):
    #     assert buffer == jnp.array([[process_id, 10 * process_id, 100 * process_id]],)
    #     assert buffer.devices() == {jax.local_devices()[id]}


main()
