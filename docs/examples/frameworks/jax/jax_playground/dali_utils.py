jpeg_path_dali_extra = "/home/ktokarski/DALI_extra/db/single/jpeg"
num_dali_extra_images = 47

import numpy as np

def get_images(device, resize=None, batch_size=num_dali_extra_images, num_batches=1, rand_res_crop=False, same=None):
    from nvidia.dali import fn, types, pipeline_def

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, seed=42)
    def pipeline():
        encoded, _ = fn.readers.file(file_root=jpeg_path_dali_extra)
        image = fn.decoders.image(encoded, device="cpu" if device == "cpu" else "mixed", output_type=types.RGB)
        if resize is not None:
            if rand_res_crop:
                image = fn.random_resized_crop(image, size=resize, seed=42)
            else:
                image = fn.resize(image, size=resize)
        if same is not None:
            image = fn.permute_batch(image, indices=[same] * batch_size)
        return image

    p = pipeline()
    p.build()
    outs = []
    for _ in range(num_batches):
        out, = p.run()
        if device == "cpu":
            outs.append([np.array(s) for s in out])
        else:
            outs.append(out)
    return outs


def get_optimized_hlo(fn, *args):
    import jax
    return jax.jit(fn).lower(*args).compile().as_text()

def get_jaxpr(fn, *args):
    import jax
    from jax import make_jaxpr
    return make_jaxpr(fn)(*args)