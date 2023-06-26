import numpy as np

from nvidia.dali import fn, types

def random_erasure_params(count=1, min_area=0.02, max_area=1 / 3, min_aspect=1 / 3, max_aspect=None,
                          seed=12345):
    rng = np.random.default_rng(seed)
    max_aspect = max_aspect or 1 / min_aspect
    log_aspect_range = np.log(min_aspect), np.log(max_aspect)

    def cb():

        def rectangle():
            area = rng.uniform(min_area, max_area) / count
            aspect_ratio = np.exp(rng.uniform(*log_aspect_range))
            shape = [aspect_ratio, 1. / aspect_ratio]
            h = np.clip(np.sqrt(area * aspect_ratio), 0, 1)
            w = np.clip(np.sqrt(area / aspect_ratio), 0, 1)
            shape = np.array([h, w], dtype=np.float32)
            anchor = np.float32(rng.uniform([0., 0.], 1. - shape))
            return shape, anchor

        params = [rectangle() for _ in range(count)]
        return tuple(np.array([param[i] for param in params]) for i in range(2))

    return cb


def random_erasure(image, prob=0.25, mode="random", count=1, min_area=0.02, max_area=1 / 3,
                   min_aspect=1 / 3, max_aspect=None, color_range=255, fill_value=0,
                   num_channels=3):

    if mode not in ("uniform", "random", "pixel"):
        raise Exception(f"Unknown random erasure mode {mode}.")

    if mode == "pixel":
        # TODO add option to fn.erase to fill the rectangles with random values
        # per each pixel
        raise NotImplemented

    if fn.random.coin_flip(probability=prob):
        shapes, anchors = fn.external_source(
            source=random_erasure_params(count, min_area, max_area, min_aspect, max_aspect),
            batch=False, num_outputs=2)

        if mode == "random":
            fill_values = fn.random.uniform(range=[0, 255], dtype=types.FLOAT, shape=(count, 3))
        else:
            fill_values = [fill_value] * count

        for i in range(count):
            image = fn.erase(image, anchor=anchors[i], shape=shapes[i], fill_value=fill_values[i],
                             normalized=True)

    return image
