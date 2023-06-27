import numpy as np

from nvidia.dali import fn, types
from nvidia.dali import math as dali_math


def mixup_cutmix_one_hot_encoding(label_smoothing, num_classes, label):
    if not label_smoothing:
        return fn.one_hot(label, num_classes=num_classes)
    else:
        off_value = label_smoothing / num_classes
        on_value = 1. - label_smoothing + off_value
        return fn.one_hot(label, off_value=off_value, on_value=on_value, num_classes=num_classes)


def mixup_cutmix(mixup_prob, mixup_switch_prob, mixup_alpha, cutmix_alpha, batch_uniform, batch_size, img_shape, image, label):
    # TODO (make fn.batch_permutation accept optional input to make it
    #       possible to generate indicies in the split batches independently)
    # TODO (make fn.multi_paste accept permuted inputs rather than indicies)
    permutation = fn.batch_permutation()
    perm_image = fn.permute_batch(image, indices=permutation)
    perm_label = fn.permute_batch(label, indices=permutation)
    cutmixed_image, cutmixed_label = cutmix(cutmix_alpha, batch_uniform, batch_size, img_shape, image, label, permutation)

    if fn.random.coin_flip(probability=mixup_prob):
        switch_to_cutmix = fn.random.coin_flip(probability=mixup_switch_prob)
        if batch_uniform:
            switch_to_cutmix = make_batch_uniform(switch_to_cutmix)
        if switch_to_cutmix:
            image, label = mixup(mixup_alpha, batch_uniform, batch_size, image, label, perm_image, perm_label)
        else:
            image, label = cutmixed_image, cutmixed_label

    return image, label


def mixup(alpha, batch_uniform, batch_size, image, label, perm_image, perm_label):
    weight = fn.external_source(source=beta_dist_weights(alpha, batch_uniform, batch_size), batch=True)
    complement = fn.cast_like(1 - weight, weight)
    mixed_image = fn.cast_like(weight * image + complement * perm_image, image)
    mixed_label = fn.cast_like(weight * label + complement * perm_label, label)
    return mixed_image, mixed_label


def cutmix(alpha, batch_uniform, batch_size, img_shape, image, label, permutation):
    weight = fn.external_source(source=beta_dist_weights(alpha, batch_uniform, batch_size), batch=True)
    sample_idx = fn.external_source(
        lambda _: np.arange(0, batch_size, dtype=np.int32),
        batch=True)
    rect_sizes, anchors = cutmix_shape(weight, img_shape)
    if batch_uniform:
        rect_sizes = make_batch_uniform(rect_sizes)
        anchors = make_batch_uniform(anchors)
    perm_label = fn.permute_batch(label, indices=permutation)
    mixed_label = fn.cast_like(weight * label + (1 - weight) * perm_label, label)
    in_ids = fn.stack(sample_idx, permutation)
    mixed_image = fn.multi_paste(
        image, in_ids=in_ids, in_anchors=anchors, out_anchors=anchors,
        shapes=rect_sizes, output_size=img_shape)
    return mixed_image, mixed_label


def cutmix_shape(weight, img_shape):
    scale = dali_math.sqrt(1 - weight)
    background_shape = types.Constant(np.array(img_shape))
    foreground_shape = scale * background_shape
    avail_space = background_shape - foreground_shape
    foreground_shape = fn.cast_like(foreground_shape, background_shape)
    rect_sizes = fn.stack(background_shape, foreground_shape)
    rand_origin_y = fn.random.uniform(
        range=fn.stack(types.Constant(0, dtype=types.FLOAT), avail_space[0]),
        dtype=types.INT32)
    rand_origin_x = fn.random.uniform(
        range=fn.stack(types.Constant(0, dtype=types.FLOAT), avail_space[1]),
        dtype=types.INT32)
    rand_origin = fn.stack(rand_origin_y, rand_origin_x)
    anchors = fn.stack(types.Constant(np.array([0, 0], dtype=np.int32)), rand_origin)
    return rect_sizes, anchors


def make_batch_uniform(sample):
    # TODO, that's ugly hack to get around Scalar optimization in types.Constant
    first_idx = types.Constant(np.array([0], dtype=np.int32))[0]
    return fn.permute_batch(sample, indices=first_idx)


def beta_dist_weights(alpha, batch_uniform, batch_size, seed=12345):

    rng = np.random.default_rng(seed)

    if not batch_uniform:
        def cb(_):
            return np.float32(rng.beta(alpha, alpha, batch_size))
    else:
        def cb(_):
            w = rng.beta(alpha, alpha)
            return np.array([w] * batch_size, dtype=np.float32)

    return cb
