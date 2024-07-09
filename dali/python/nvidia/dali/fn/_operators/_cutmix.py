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

from typing import Optional, Tuple, Union

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import fn, types, math as dmath
import nvidia.dali.ops._python_def_op_utils as _python_def_op_utils


def _get_bbox(lamb, margin=None):
    # lambd = 1 => so the background should take it all
    # side = sqrt(1 - 1) = 0
    # tl_anchor => [0, 1], shape = 0

    # lambd = 0 => so the background should be covered totally
    # side = 1
    # tl_anchor = (0, 0), shape = (1, 1)
    side = dmath.sqrt(1 - lamb)
    print("margin dddd", margin)
    if margin == "fit":
        print("hmmst")
        tl_anchor = fn.random.uniform(range=[0, 1], shape=(2,)) * (1 - side)
        shape = fn.stack(side, side)
        actual_lamb = lamb
    else:
        radious = 0.5 * side
        if margin is None:
            middle_anchor = fn.random.uniform(range=[0, 1], shape=(2,))
        else:
            assert isinstance(margin, (_DataNode, float))
            if isinstance(margin, float):
                if not 0 <= margin < 0.5:
                    raise ValueError("The `margin` argument must be a float in [0, 0.5) range")
            middle_anchor = fn.random.uniform(range=[0, 1], shape=(2,)) * (1 - 2 * margin) + margin
        tl_anchor = dmath.clamp(middle_anchor - radious, 0, 1)
        br_anchor = dmath.clamp(middle_anchor + radious, 0, 1)
        shape = br_anchor - tl_anchor
        actual_lamb = shape[0] * shape[1]
    return actual_lamb, tl_anchor, shape


def _one_hot_encode_labels(labels, num_classes, label_smoothing, permutation):
    if permutation is not None:
        if not isinstance(labels, _DataNode):
            raise ValueError(
                f"When `samples` is a single batch, the `labels` "
                f"must be a single DALI batch (DataNode). Got {labels}"
            )
        if not isinstance(num_classes, (int, _DataNode)):
            raise ValueError(
                f"When `samples` is a single batch, the `num_classes` "
                f"must be a single int or DALI batch (DataNode). Got {labels}"
            )
        labels_0 = labels
        labels_1 = fn.permute_batch(labels, indices=permutation)
        num_classes_0, num_classes_1 = num_classes, num_classes
    else:
        if (
            not isinstance(labels, (tuple, list))
            or len(labels) != 2
            or any(not isinstance(s, _DataNode) for s in labels)
        ):
            raise ValueError(
                f"When `samples` is a pair of two batches, the `labels` "
                f"must be a pair of batches (DataNodes) too. Got {labels}"
            )
        labels_0, labels_1 = labels
        if isinstance(num_classes, int):
            num_classes_0, num_classes_1 = num_classes, num_classes
        else:
            if (
                not isinstance(num_classes, (tuple, list))
                or len(num_classes) != 2
                or any(not isinstance(s, (int, _DataNode)) for s in num_classes)
            ):
                raise ValueError(
                    f"When `samples` is a pair of two batches, the `num_classes` "
                    f"must be a pair of integers or pair of batches (DataNodes). Got {labels}"
                )
            num_classes_0, num_classes_1 = num_classes
    one_hot_encoded_0 = fn.one_hot(labels_0, num_classes=num_classes_0)
    one_hot_encoded_1 = fn.one_hot(labels_1, num_classes=num_classes_1)
    return one_hot_encoded_0, one_hot_encoded_1


def _get_one_hot_encoded_labels(labels, num_classes, one_hot_labels, label_smoothing, permutation):
    if labels is not None or num_classes is not None:
        if labels is None or num_classes is None:
            raise ValueError(
                "The `labels` and `num_classes` must be specified together, so that DALI can "
                "one-hot encode the labels and mix the encodings. If you wish to pass "
                "already one-hot encoded labels, please pass them as `one_hot_labels` instead."
            )
        if one_hot_labels is not None:
            raise ValueError("The `one_hot_labels` and `labels` cannot be specified together.")

        one_hot_labels_0, one_hot_labels_1 = _one_hot_encode_labels(
            labels, num_classes, label_smoothing, permutation
        )

    elif one_hot_labels is not None:

        if permutation is not None:
            if not isinstance(one_hot_labels, _DataNode):
                raise ValueError(
                    f"When `samples` is a single batch, the `one_hot_labels` "
                    f"must be a single DALI batch (DataNode) too. Got {one_hot_labels}"
                )
            one_hot_labels_0 = one_hot_labels
            one_hot_labels_1 = fn.permute_batch(one_hot_labels, indices=permutation)
        else:
            if (
                not isinstance(one_hot_labels, (tuple, list))
                or len(one_hot_labels) != 2
                or any(not isinstance(s, _DataNode) for s in one_hot_labels)
            ):
                raise ValueError(
                    f"When `samples` is a pair of two batches, the `one_hot_labels` "
                    f"must be a pair of batches (DataNodes) too. Got {one_hot_labels}"
                )
            one_hot_labels_0, one_hot_labels_1 = one_hot_labels
    else:
        one_hot_labels_0, one_hot_labels_1 = None, None

    return one_hot_labels_0, one_hot_labels_1


def _cutmix_samples(samples_0, samples_1, anchor, shape, preserve):
    anchors_rel = fn.stack(types.Constant([0, 0], dtype=types.DALIDataType.FLOAT), anchor)
    shapes_rel = fn.stack(types.Constant([1, 1], dtype=types.DALIDataType.FLOAT), shape)
    return fn.multi_paste(
        samples_0,
        samples_1,
        in_anchors_rel=anchors_rel,
        out_anchors_rel=anchors_rel,
        shapes_rel=shapes_rel,
        preserve=preserve,
    ), anchors_rel, shapes_rel


def _cutmix_labels(one_hot_labels_0, one_hot_labels_1, actual_lamb):
    if one_hot_labels_0 is None:
        return None
    else:
        return actual_lamb * one_hot_labels_0 + (1 - actual_lamb) * one_hot_labels_1


def _get_samples(samples, seed=None):
    if isinstance(samples, (tuple, list)):
        if not len(samples) == 2:
            raise ValueError(
                f"The `samples` must be a single DALI batch (DataNode) or a "
                f"tuple of excatly two batches. Got a tuple/list of length {len(samples)}."
            )
        perm = None
        samples_0, samples_1 = samples

        for s, ord in ((samples_0, "first"), (samples_1, "second")):
            if not isinstance(s, _DataNode):
                raise ValueError(
                    f"The `samples` must be a single DALI batch (DataNode) or a tuple of "
                    f"excatly two batches. However, the {ord} element of the tuple "
                    f"is not a DataNode (got {s})."
                )
    else:
        samples_0, samples_1 = samples, None

        if not isinstance(samples_0, _DataNode):
            raise ValueError(
                f"The `samples` must be a single DALI batch (DataNode) or a "
                f"tuple of excatly two batches. Got {samples}."
            )

        perm = fn.batch_permutation(seed=seed, no_fixed_points=True)
        samples_1 = fn.permute_batch(samples, indices=perm)

    return samples_0, samples_1, perm


def _get_lambda(
    alpha: Optional[Union[float, _DataNode]] = 1.0,
    lamb: Optional[Union[float, _DataNode]] = None,
    seed: Optional[int] = None,
):
    if lamb is None:
        lamb = fn.random.beta(alpha=alpha, beta=alpha, seed=seed)
    elif isinstance(lamb, float):
        if not 0 <= lamb <= 1:
            raise ValueError(
                f"The lambda parameter must be a float in [0, 1] range describing the "
                f"proportion of areas of mixed images. Got {lamb}."
            )
    else:
        if not isinstance(lamb, _DataNode):
            raise ValueError(
                f"The lambda parameter must be a single float in [0, 1] range or a "
                f"DALI node (output of DALI operators or types.Constant) representing "
                f"a batch of floats. Got {lamb}."
            )
    return lamb


def cutmix(
    samples: Union[_DataNode, Tuple[_DataNode, _DataNode]],
    labels: Optional[Union[_DataNode, Tuple[_DataNode, _DataNode]]] = None,
    num_classes: Optional[int] = None,
    label_smoothing: Optional[float] = None,
    one_hot_labels: Optional[Union[_DataNode, Tuple[_DataNode, _DataNode]]] = None,
    alpha: Optional[Union[float, _DataNode]] = 1.0,
    lamb: Optional[Union[float, _DataNode]] = None,
    margin: Optional[Union[str, float, _DataNode]] = None,
    preserve: bool = False,
    seed: Optional[int] = None,
):
    """
    Applies `CutMix <https://arxiv.org/abs/1905.04899>` augmentation to a batch of
    images or videos.

    If ``samples`` is a pair of two batches, the i-th output sample is formed by
    replacement of a random rectangular region in the i-th sample from the first batch
    with a rectangular patch from the i-th sample in the second batch.
    If `samples` is a single batch, the batch is mixed with the permutation of the same batch.

    In either case, the samples across all the batches must have uniform shapes.

    By default, the area of the patched region is controlled by ``alpha`` paramter. A ``λ``
    parameter is sampled from the :func:`Beta(alpha, alpha)<nvidia.dali.fn.random.beta>`
    distribution and the patch is a rectangle of size

    .. math:: (\sqrt{1-\lambda} \cdot H, \sqrt{1-\lambda} \cdot W)

    Alternatively, you can specify ``lamb`` parameter, in that case it will be used directly
    as the ``λ``.

    In either case, the actual size of the patch may be smaller, because it is clamped
    when it does not fully fit into output when randomly placed over the first sample.
    The bahviour can be adjusted with ``margin`` parameter.

    If provided with labels, cutmix additonally outputs a weighted mean of one-hot
    encoded labels were the weights correspond to the are of the mixed samples. I.e.:

    .. math:: \lambda' \cdot a + (1-\lambda') \cdot b

    where ``a`` and ``b`` are one-hot encoded labels and the ``λ'`` is ``λ`` adjusted
    for clampping of the replaced region.

    The integer labels can be passed as ``labels`` to be one-hot encoded first or
    the already encoded labels can be passed as ``one_hot_labels`` directly.

    Args
    ----
    samples : DataNode or a pair of DataNodes
        Either a batch or a batch of images or videos. The samples in the batch or across
        both batches must have uniform shape. If two batches are provided, the output samples
        are result of sample-wise mixing of the two batches.

        If a single batch is provided, it is sample-wise mixed with its permutation.
        In that case, the output dictionary will contain a ``permutation`` item, describing
        the permutation and the ``permuted_samples``, the batch of permuted samples according
        to ``permutation``.
    labels : DataNode or tuple of DataNodes, optional, default None
        If specified, it must be a batch or a pair of batches with integer labels corresponding
        to the ``samples``. It must be a pair of batches iff the ``samples`` is a pair of batches.
        It must be specified together with ``num_classes`` and, optionally, ``label_smoothing``,
        so that the ``labels`` can be one-hot encoded. A batch of the encoded, mixed labels
        for corresponding output samples is returned as the ``mixed_labels`` item in the output
        dictionary.
    num_classes : int, optional, default None
        The number of classes in the dataset that the ``samples`` come from.
        It must be specified alongisde ``labels`` parameter, so that the ``labels``
        can be one-hot encoded.
    label_smoothing : float, optional, default None
        An optional label smoothing factor used in one-hot encoding of the ``labels``
    one_hot_labels : DataNode or pair of DataNodes, optional, default None
        Already encoded labels, to be used directly in the weighted sum.
        If specified, the ``labels``, ``num_classes``, and ``label_smoothing`` must be None.
        A batch of the mixed labels for corresponding output samples is returned as
        the ``mixed_labels`` item in the output dictionary.
    alpha: float or DataNode, optional, default 1
        The parameter of the :func:`Beta(alpha, alpha)<nvidia.dali.fn.random.beta>`
        distribution that is used to sample area of the patched region as follows:

        The parameter ``lambda`` is sampled from the distribution

        .. math:: \lambda ~ \Beta(\alpha, \alpha)

        and the size of the patch is

        .. math:: (\sqrt{1-\lambda} \cdot H, \sqrt{1-\lambda} \cdot W)

        The actual size of the patch may be smaller if the patch protrudes
        from the output sample, due to random selection of its origin in the output
        canvas. That behaviour can be controlled with ``marigin`` parameter.

        The randomly sampled and the actual ``lambda`` corrected for the protruding
        patch are returned in the output dictionary as ``lambda`` and ``lambda_corrected``
        respectively.

        If ``lamb`` is specified, the ``alpha`` parameter is ignored.
    lamb: float or DataNode, optional, default None
        If specified, it must be a float in [0, 1] range. Then, the shape of the
        patch is

        .. math:: (\sqrt{1-\lamb} \cdot H, \sqrt{1-\lamb} \cdot W)

        The actual size of the patch may be smaller if the patch protrudes
        from the output sample, due to random selection of its origin in the output
        canvas. That behaviour can be controlled with ``marigin`` parameter.

        The ``lambda`` corrected for the protruding patch is returned in the output
        dictionary as ``lambda_corrected`` item.

        If ``lamb`` is specified, the ``alpha`` parameter is ignored.

    margin: str, float, optional, default None
        Controls how the a patch is randomly placed over the output sample.

        By default, the patch is anchored at its center and the position for
        the anchor in the output is sampled uniformly at random. This way,
        the patch may not fit into the output sample, and the protruding
        part will be discarded, making the actual area of the patch
        smaller than the ``1-lambda``.

        The ``lambda`` corrected for the protruding patch is returned in the output
        dictionary as ``lambda_corrected`` item.

        Alternatively, you can specify ``margin="fit"``, in that case the
        position in the output is sampled excluding margins, so that the
        patch always fully fits inside the output sample.

        If ``margin`` is a float, it must be in ``[0, 0.25)`` range, in that case,
        the position for the patch center is sampled from (margin * E, (1 - margin) * E)
        for ``E=height`` and ``E=width`` respectively.

    preserve: bool, default=True
        If set to False, the returned DALI function may be optimized out of the DALI pipeline,
        if it does not return any outputs or none of the function outputs contribute
        to the pipeline's output.

    Returns
    -------
    Tuple[_DataNode, Dict]
        The first element is a batch of mixed samples, the second is a dictionary
        containing additional outputs. The exact items presence depends on the
        specified arguments.
    """
    lamb = _get_lambda(alpha, lamb, seed=seed)
    samples_0, samples_1, permutation = _get_samples(samples, seed=seed)
    actual_lamb, tl_anchor, shape = _get_bbox(lamb, margin=margin)
    cutmixed, anchors_rel, shapes_rel = _cutmix_samples(samples_0, samples_1, tl_anchor, shape, preserve=preserve)
    one_hot_labels_0, one_hot_labels_1 = _get_one_hot_encoded_labels(
        labels,
        num_classes,
        one_hot_labels,
        label_smoothing,
        permutation=permutation,
    )
    cutmixed_labels = _cutmix_labels(one_hot_labels_0, one_hot_labels_1, actual_lamb)
    return cutmixed, {
        "mixed_labels": cutmixed_labels,
        "permutation": permutation,
        "permuted_samples": samples_1,
        "lambda_corrected": actual_lamb,
        "lambda": lamb,
        "tl_anchor": tl_anchor, 
        "shape": shape,
        "anchors_rel": anchors_rel,
        "shapes_rel": shapes_rel,
    }


_cutmix_function_desc = _python_def_op_utils.PyOpDesc(
    "nvidia.dali.fn",
    "cutmix",
    ["CPU", "GPU"],
    "Applies cutmix transformation to a batch of images or videos",
)
