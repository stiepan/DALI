import math

import numpy as np

from nvidia.dali import fn, types
from nvidia.dali import math as dali_math


def random_erasure_params(count=1, min_area=0.02, max_area=1 / 3, min_aspect=1 / 3,
                          max_aspect=None):
    max_aspect = max_aspect or 1 / min_aspect
    min_aspect_log = math.log(min_aspect)
    max_aspect_log = math.log(max_aspect)

    def rectangle():
        area = fn.random.uniform(range=[min_area, max_area]) / count
        aspect_ratio = fn.random.uniform(range=[min_aspect_log, max_aspect_log])
        aspect_ratio = dali_math.exp(aspect_ratio)
        shape = fn.stack(aspect_ratio, 1 / aspect_ratio)  # [aspect_ratio, 1 / aspect_ratio]
        shape = dali_math.sqrt(area * shape)
        shape = dali_math.clamp(shape, 0, 1)
        anchor_bound = np.array(1., dtype=np.float32) - shape
        anchor_y_range = fn.stack(np.array(0., dtype=np.float32), anchor_bound[0])
        anchor_x_range = fn.stack(np.array(0., dtype=np.float32), anchor_bound[1])
        anchor_y = fn.random.uniform(range=anchor_y_range)
        anchor_x = fn.random.uniform(range=anchor_x_range)
        anchor = fn.stack(anchor_y, anchor_x)
        return shape, anchor

    rectangles = [rectangle() for _ in range(count)]
    shapes = [shape for shape, _ in rectangles]
    anchors = [anchor for _, anchor in rectangles]

    return fn.stack(*shapes), fn.stack(*anchors)


def random_erasure(image, prob=0.25, mode="random", count=1, min_area=0.02, max_area=1 / 3,
                   min_aspect=1 / 3, max_aspect=None, color_range=255, fill_value=0,
                   num_channels=3):

    if mode not in ("uniform", "random", "pixel"):
        raise Exception(f"Unknown random erasure mode {mode}.")

    if mode == "pixel":
        # TODO add option to fn.erase to fill the rectangles with random values
        # per each pixel
        raise NotImplemented

    max_aspect = max_aspect or 1 / min_aspect

    if fn.random.coin_flip(probability=prob):
        shapes, anchors = random_erasure_params(count, min_area, max_area, min_aspect, max_aspect)

        if mode == "random":
            fill_values = fn.random.uniform(range=[0, color_range], dtype=types.FLOAT,
                                            shape=(count, num_channels))
        else:
            fill_values = [fill_value] * count

        for i in range(count):
            image = fn.erase(image, anchor=anchors[i], shape=shapes[i], fill_value=fill_values[i],
                             normalized=True)

    return image
