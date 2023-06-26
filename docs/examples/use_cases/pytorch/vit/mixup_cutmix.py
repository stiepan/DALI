import numpy as np

from nvidia.dali import fn


def mixup_cutmix_one_hot_encoding(label_smoothing, num_classes, label):
    if not label_smoothing:
        return fn.one_hot(label, num_classes=num_classes)
    else:
        off_value = label_smoothing / num_classes
        on_value = 1. - label_smoothing + off_value
        return fn.one_hot(label, off_value=off_value, on_value=on_value, num_classes=num_classes)


def mixup_cutmix(mixup_prob, mixup_switch_prob, mixup_alpha, cutmix_alpha, batch_uniform,
                 batch_size, img_shape, image, label):
    # TODO (make fn.batch_permutation accept optional input to make it
    #       possible to generate indicies in the split batches independently)
    # TODO (make fn.multi_paste accept permuted inputs rather than indicies)
    permutation = fn.batch_permutation()
    perm_image = fn.permute_batch(image, indices=permutation)
    perm_label = fn.permute_batch(label, indices=permutation)
    mixed_image, mixed_label = mixup(mixup_alpha, batch_uniform, batch_size, image, label,
                                     perm_image, perm_label)
    cutmixed_image, cutmixed_label = cutmix(cutmix_alpha, batch_uniform, batch_size, img_shape,
                                            image, label, permutation)

    if fn.random.coin_flip(probability=mixup_prob):
        if not batch_uniform:
            switch_to_cutmix = fn.random.coin_flip(probability=mixup_switch_prob)
        else:
            switch_to_cutmix = fn.external_source(
                source=uniform_coin_flip(mixup_switch_prob, batch_size), batch=True)
        if switch_to_cutmix:
            image, label = mixed_image, mixed_label
        else:
            image, label = cutmixed_image, cutmixed_label

    return image, label


def mixup(alpha, batch_uniform, batch_size, image, label, perm_image, perm_label):
    weight = fn.external_source(source=mix_up_weights(alpha, batch_uniform, batch_size), batch=True)
    mixed_image = fn.cast_like(weight * image + (1 - weight) * perm_image, image)
    mixed_label = fn.cast_like(weight * label + (1 - weight) * perm_label, label)
    return mixed_image, mixed_label


def cutmix(alpha, batch_uniform, batch_size, img_shape, image, label, permutation):
    weight, rect_size, anchor = fn.external_source(
        source=cutmix_shape(alpha, batch_uniform, batch_size, img_shape), num_outputs=3, batch=True)
    sample_idx = fn.external_source(
        lambda source_info: np.array(source_info.idx_in_batch, dtype=np.int32), batch=False)
    perm_label = fn.permute_batch(label, indices=permutation)
    mixed_label = fn.cast_like(weight * label + (1 - weight) * perm_label, label)
    in_ids = fn.stack(sample_idx, permutation)
    mixed_image = fn.multi_paste(image, in_ids=in_ids, in_anchors=anchor, out_anchors=anchor,
                                 shapes=rect_size, output_size=img_shape)
    return mixed_image, mixed_label


def mix_up_weights(alpha, batch_uniform, batch_size, seed=12345):

    rng = np.random.default_rng(seed)

    def cb():
        if not batch_uniform:
            return np.float32(rng.beta(alpha, alpha, batch_size))
        else:
            w = rng.beta(alpha, alpha)
            return np.array([w] * batch_size, dtype=np.float32)

    return cb


# workaround for uniform coin flip accross the whole batch
def uniform_coin_flip(mixup_switch_prob, batch_size, seed=12345):

    rng = np.random.default_rng(seed)

    def cb():
        v = rng.uniform(0, 1)
        return np.array([v <= mixup_switch_prob] * batch_size)

    return cb


def cutmix_shape(alpha, batch_uniform, batch_size, img_shape, seed=12345):
    rng = np.random.default_rng(seed)

    # TODO consider relaxing the rand_origin bounds - allow in-pasted image to not fit perfectly
    # (which implicitly reduces the scale)
    def single_sample_params():
        weight = rng.beta(alpha, alpha)
        scale = np.sqrt(1 - weight)
        background_shape = np.array(img_shape)
        foreground_shape = scale * background_shape
        rect_sizes = np.array([background_shape, foreground_shape], dtype=np.int32)
        rand_origin = rng.uniform([0, 0], background_shape - foreground_shape)
        anchor = np.array([[0, 0], rand_origin], dtype=np.int32)
        return np.array(weight), rect_sizes, anchor

    def cb():
        if batch_uniform:
            weight, ract_size, anchor = single_sample_params
            return tuple(np.array([param] * batch_size) for param in single_sample_params)
        else:
            params = [single_sample_params() for _ in range(batch_size)]
            return tuple(np.array([t[i] for t in params]) for i in range(3))

    return cb
