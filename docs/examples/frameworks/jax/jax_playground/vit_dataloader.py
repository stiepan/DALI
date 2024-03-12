import numpy as np
import jax
import jax.numpy as jnp
import jax.dlpack as jpack
import dm_pix as pix

from nvidia.dali import fn, types, pipeline_def
# from nvidia.dali.python_function_plugin import current_dali_stream

from jax_playground.dali_utils import get_images

batch_size = 32

source_images_gpu = get_images("gpu", (384, 384), batch_size, 2, rand_res_crop=True, same=5)

# img = fn.random_resized_crop(img, size=self.image_shape[:-1], seed=self.seed)
# img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip(seed=self.seed))

# ## color jitter
# brightness = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
# contrast = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
# saturation = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
# hue = fn.random.uniform(range=[0.9,1.1], seed=self.seed)
# img = fn.color_twist(img,
#                         brightness=brightness,
#                         contrast=contrast,
#                         hue=hue,
#                         saturation=saturation)

# ## auto-augment
# ## `shape` controls the magnitude of the translation operations
# img = auto_augment.auto_augment_image_net(img, seed=self.seed)


def jax_random_flip_horz(key, image):
    return pix.random_flip_left_right(key, image)


def hue_mat(hue):
    h_rad = hue * jnp.pi / 180
    ret = jnp.eye(3)
    ret = ret.at[1, 1].set(jnp.cos(h_rad))
    ret = ret.at[2, 2].set(jnp.cos(h_rad))
    ret = ret.at[1, 2].set(jnp.sin(h_rad))
    ret = ret.at[2, 1].set(-jnp.sin(h_rad))
    return ret


def sat_mat(sat):
    ret = jnp.eye(3)
    ret = ret.at[1, 1].set(sat)
    ret = ret.at[2, 2].set(sat)
    return ret


def eye3(val):
    return jnp.diag(jnp.full(3, val, dtype=jnp.float32))


def color_twist_mat(brightness, contrast, hue, saturation, value):
    rgb2yiq = jnp.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.321],
        [0.211, -0.523, 0.311]
    ])
    yiq2rgb = jnp.linalg.inv(rgb2yiq)
    return eye3(brightness) @ eye3(contrast) @ yiq2rgb @ hue_mat(hue) @ sat_mat(saturation) @ eye3(value) @ rgb2yiq


def jax_color_twist(image, brightness, contrast, saturation, hue, value):
    mat = color_twist_mat(brightness, contrast, saturation, hue, value)
    dtype = image.dtype
    dtype_info = jnp.iinfo(dtype)
    image = jnp.clip(image @ mat, dtype_info.min, dtype_info.max)
    return jnp.asarray(image, dtype=dtype)


def jax_random_color_twist(key, image):
    keys = jax.random.split(key, num=4)
    brightness = jax.random.uniform(keys[0], (), minval=0.6, maxval=1.4)
    contrast = jax.random.uniform(keys[1], (), minval=0.6, maxval=1.4)
    saturation = jax.random.uniform(keys[2], (), minval=0.6, maxval=1.4)
    hue = jax.random.uniform(keys[3], (), minval=0.9, maxval=1.1)
    return jax_color_twist(image, brightness, contrast, saturation, hue, 1)


def jax_augmentations_impl(key, image):
    key, sub_key = jax.random.split(key)
    image = jax_random_flip_horz(sub_key, image)
    key, sub_key = jax.random.split(key)
    image = jax_random_color_twist(sub_key, image)
    return image


def batch_adapter(sample_cb, device):

    batched_cb = jax.vmap(sample_cb)

    def inner(key, batch):
        n_samples = batch.shape[0]
        keys = jax.random.split(key, num=n_samples)
        return batched_cb(keys, batch)

    return jax.jit(inner, device=device)

class JaxAugmentations:
    def __init__(self, sample_cb, seed=42, device_id=0):
        gpus = jax.devices("gpu")
        assert len(gpus) > device_id
        self.key = jax.random.PRNGKey(seed)
        self.sample_cb = sample_cb
        # self.batched_cb = jax.jit(jax.vmap(sample_cb), device=gpus[device_id])
        self.batched_cb = batch_adapter(sample_cb, gpus[device_id])

    # def inner(self, key, batched_cb, images):
        # print(dir(images))
        # with jax.transfer_guard("disallow"):
        # jmages = [jpack.from_dlpack(image) for image in images]
        # jbatch = jnp.stack(jmages)
        # jout_batch = batched_cb(key, jbatch)
        # out_batch = [jpack.to_dlpack(sample, stream=stream) for sample in jout_batch]
        # out_batch = jpack.to_dlpack(jout_batch, stream=stream)
        # return out_batch

    def __call__(self, images):
        # TODO is it safe to split in tree-like fashion, or does it need to be linear
        # with the last key as the source for the next iters?
        # dali_stream = current_dali_stream()
        # self.key, sub_key = jax.random.split(self.key)
        return self.batched_cb(self.key, images)


@pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, prefetch_queue_depth=4)
def jaxline():
    # image = fn.external_source(
    #     lambda i: source_images_gpu[i % len(source_images_gpu)],
    #     batch=True, no_copy=True, device="gpu",
    #     layout="HWC", dtype=types.UINT8)
    image = types.Constant([np.array(s) for s in source_images_gpu[0].as_cpu()][5], device="gpu")
    # if fn.random.coin_flip():
    #     image = image
    image = fn.dl_tensor_python_function(
        image,
        batch_processing=True,
        output_layouts="HWC",
        function=JaxAugmentations(jax_augmentations_impl),
        synchronize_stream=False,
        dl_tensor_stream_aware=True,
    )
    return image


@pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, prefetch_queue_depth=4)
def dali_pipeline():
    # image = fn.external_source(
    #     lambda i: source_images_gpu[i % len(source_images_gpu)],
    #     batch=True, no_copy=True, device="gpu",
    #     layout="HWC", dtype=types.UINT8)

    image = types.Constant([np.array(s) for s in source_images_gpu[0].as_cpu()][5], device="gpu")
    image = fn.flip(image, depthwise=0, horizontal=fn.random.coin_flip(seed=42))

    # color jitter
    # brightness = fn.random.uniform(range=[0.6,1.4], seed=42)
    # contrast = fn.random.uniform(range=[0.6,1.4], seed=42)
    # saturation = fn.random.uniform(range=[0.6,1.4], seed=42)
    # hue = fn.random.uniform(range=[0.9,1.1], seed=42)
    # image = fn.color_twist(image,
    #                         brightness=brightness,
    #                         contrast=contrast,
    #                         hue=hue,
    #                         saturation=saturation)
    return image
