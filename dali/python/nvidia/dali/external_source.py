# custom wrappers around ops
from nvidia.dali import backend as _b
import inspect
import multiprocessing
from multiprocessing import Process, Pool, cpu_count


def _get_batch_shape(data):
    if isinstance(data, (list, tuple, _b.TensorListCPU, _b.TensorListGPU)):
        if len(data) == 0:
            return [], True
        if callable(data[0].shape):
            return [x.shape() for x in data], False
        else:
            return [x.shape for x in data], False
    else:
        shape = data.shape
        if callable(shape):
            shape = data.shape()
        return [shape[1:]] * shape[0], True

def _check_data_batch(data, batch_size, layout):
    shape, uniform = _get_batch_shape(data)
    if len(shape) != batch_size:
        raise RuntimeError("The external source callback returned an unexpected batch "
        "size: {} instead of {}".format(len(shape), batch_size))

    if len(shape) > 0:
        dim = len(shape[0])
        if not uniform:
            for ts in shape:
                if len(ts) != dim:
                    raise RuntimeError("All tensors in a batch must have the same number of dimensions")
        if layout is not None and layout != "" and dim != len(layout):
            raise RuntimeError("The layout '{}' cannot describe {}-dimensional data".format(layout, dim))

class _CycleIter():
    def __init__(self, iterable):
        self.source = iterable

    def __iter__(self):
        self.it = iter(self.source)
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source)
            return next(self.it)

class _CycleGenFunc():
    def __init__(self, gen_func):
        self.source = gen_func

    def __iter__(self):
        self.it = iter(self.source())
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source())
            return next(self.it)

class DamtaSemmmt(object):

    class Count(object):
        co_argcount = 1

    def __init__(self):
        self.__code__ = self.Count()


def worker(proc_id, callback, task_queue, res_queue):
    print("Worker {} starts".format(proc_id))
    while True:
        idx = task_queue.get()
        # print("Worker {} got task {}".format(proc_id, idx))
        if (idx < 0):
            break
        res = callback(idx)
        # print("Worker {} puts task {}".format(proc_id, idx))
        res_queue.put((idx, res))
        # print("Worker {} put task {}".format(proc_id, idx))


class WorkersPool(object):

    def __init__(self, callback, workers_no=None):
        mp = multiprocessing.get_context("fork")
        self.round_counter = 0
        self.workers_no = workers_no if workers_no is not None else multiprocessing.cpu_count()
        self.processes = []
        self.task_queues = []
        self.res_queue = mp.Queue()
        for i in range(self.workers_no):
            task_queue = mp.Queue()
            process = mp.Process(
                target=worker,
                args=(i, callback, task_queue, self.res_queue),
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for p in self.processes:
            p.join()

    def next_worker(self):
        counter = self.round_counter
        self.round_counter = (self.round_counter + 1) % self.workers_no
        return counter

    def process_batch(self, tasks):
        split_tasks = [(task_id, self.next_worker()) for task_id in tasks]
        for task_id, worker_id in split_tasks:
            self.task_queues[worker_id].put(task_id)
        done_tasks = dict(self.res_queue.get() for _ in range(len(split_tasks)))
        return [done_tasks[task_id] for task_id, _ in split_tasks]


class _ExternalSourceGroup(object):
    def __init__(self, callback, is_multioutput, instances = [], cuda_stream = None, use_copy_kernel = None, batch_size=None):
        # print("How many times am I hapenning, hmm?")
        self.instances = list(instances)  # we need a copy!
        self.is_multioutput = is_multioutput
        self.callback = callback
        self._cuda_stream = cuda_stream
        self._batch_size = batch_size
        self.use_copy_kernel = use_copy_kernel
        if callback is not None:
            if callback.__code__.co_argcount not in [0, 1, 2]:
                raise TypeError("External source callback must be a callable with 0 - 2 arguments")
            self.accepts_iter_num = (callback.__code__.co_argcount > 0)
        else:
            self.accepts_iter_num = None
        if not batch_size:
            self.pool = None
        else:
            self.pool = WorkersPool(callback)

    def append(self, instance):
        self.instances.append(instance)

    def call_and_feed(self, pipeline, current_iter):
        # print('ddddd', current_iter)
        if self._batch_size:
            batch_stride = current_iter * self._batch_size
            callback_out = self.pool.process_batch(range(batch_stride, batch_stride + self._batch_size))
            if self.is_multioutput:
                callback_out = [list(l) for l in zip(*callback_out)]
                # print(len(callback_out))

        elif self.accepts_iter_num:
            callback_out = self.callback(current_iter)
        else:
            callback_out = self.callback()

        if self.is_multioutput:
            for op in self.instances:
                data = callback_out[op._output_index]
                pipeline.feed_input(op._name, data, op._layout, self._cuda_stream, self.use_copy_kernel)
        else:
            data = callback_out
            op = self.instances[0]
            pipeline.feed_input(op._name, data, op._layout, self._cuda_stream, self.use_copy_kernel)

def _is_generator_function(x):
    """Checks whether x is a generator function or a callable object
    where __call__ is a generator function"""
    if inspect.isgeneratorfunction(x):
        return True
    if x is None or inspect.isfunction(x):
        return False
    if isinstance(x, DamtaSemmmt):
        return False
    call = getattr(x, "__call__", None)
    return _is_generator_function(call)

def _get_callback_from_source(source, cycle):
    iterable = False
    if source is not None:
        try:
            if cycle:
                if inspect.isgenerator(source):
                    raise TypeError("Cannot cycle through a generator - if the generator is a result "
                        "of calling a generator function, pass that function instead as `source`.")
                if _is_generator_function(source):
                    iterator = iter(_CycleGenFunc(source))
                else:
                    iterator = iter(_CycleIter(source))
            else:
                if _is_generator_function(source):
                    source = source()
                iterator = iter(source)
            iterable = True
            callback = lambda: next(iterator)
        except TypeError as err:
            if "not iterable" not in str(err):
                raise(err)
            if cycle is not None:
                raise ValueError("The argument `cycle` can only be specified if `source` is iterable")
            if not callable(source):
                raise TypeError("Source must be callable, iterable or a parameterless generator function")
            callback = source
    else:
        callback = None

    if not iterable and cycle:
        raise ValueError("`cycle` argument is only valid for iterable `source`")
    return callback

class ExternalSource():
    """ExternalSource is a special operator that can provide data to a DALI pipeline
from Python by several methods.

The simplest and preferred way is to specify a ``source``, which can be a callable or iterable.

.. warning::
    :meth:`nvidia.dali.ops.ExternalSource` operator is not compatible with TensorFlow integration.

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
"""

    _args_doc = """
Args
----

`source` : callable or iterable
    The source of the data.

    The source is polled for data (via a call ``source()`` or ``next(source)``)
    when the pipeline needs input for the next iteration. Depending on the value of ``num_outputs``,
    the source can supply one or more data batches. If ``num_outputs`` is not set, the ``source``
    is expected to return one batch. If this value is specified, the data is expected to a be tuple,
    or list, where each element corresponds to respective return value of the external_source.
    If the source is a callable that accepts a positional argument, it is assumed to be the current
    iteration number and consecutive calls will be ``source(0)``, ``source(1)``, and so on.
    If the source is a generator function, the function is invoked and treated as an iterable.
    However, unlike a generator, the function can be used with ``cycle``. In this case, the function
    will be called again when the generator reaches the end of iteration.

    For the GPU input, it is a user's responsibility to modify the provided GPU memory content
    only in the provided stream. DALI schedules a copy on this stream, and all work is properly
    queued. If no stream is provided, DALI will use a default, with a best-effort approach at
    correctness. See the ``cuda_stream`` argument documentation for more information.
    The data batch produced by ``source`` may be anything that's accepted by
    :meth:`nvidia.dali.pipeline.Pipeline.feed_input`

`num_outputs` : int, optional
    If specified, denotes the number of TensorLists that are produced by the source function.

    If set, the operator returns a list of ``DataNode`` objects, otherwise a single ``DataNode``
    object is returned.

Keyword Args
------------

`cycle`: bool
    If set to True, the source will be wrapped.

    If set to False, StopIteration is raised when the end of data is reached.
    This flag requires that the ``source`` is a collection, for example, an iterable object where
    ``iter(source)`` returns a fresh iterator on each call or a gensource erator function.
    In the latter case, the generator function is called again when more data than was
    yielded by the function is requested.

`name` : str, optional
    The name of the data node.

    Used when feeding the data in ``iter_setup`` and can be omitted if
    the data is provided by ``source``.


`layout` : :ref:`layout str<layout_str_doc>` or list/tuple thereof
    If provided, sets the layout of the data.

    When ``num_outputs > 1``, the layout can be a list that contains a distinct layout for each
    output. If the list has fewer than ``num_outputs`` elements, only the first
    outputs have the layout set, the rest of the outputs don't have a layout set.

`cuda_stream` : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``, such as ``cupy.cuda.Stream`` or ``torch.cuda.Stream``
    The CUDA stream is used to copy data to the GPU or from a GPU source.

    If this parameter is not set, a best-effort will be taken to maintain correctness. That is,
    if the data is provided as a tensor/array from a recognized library such as CuPy or PyTorch,
    the library's current stream is used. Although this approach works in typical scenarios,
    with advanced use cases, and code that uses unsupported libraries, you might need to
    explicitly supply the stream handle.

    This argument has two special values:
      * 0 - Use the default CUDA stream
      * 1 - Use DALI's internal stream

    If internal stream is used, the call to ``feed_input`` will block until the copy to internal
    buffer is complete, since there's no way to synchronize with this stream to prevent
    overwriting the array with new data in another stream.

`use_copy_kernel` : optional, `bool`
    If set to True, DALI will use a CUDA kernel to feed the data instead of cudaMemcpyAsync (default).

    .. note::
        This is applicable only when copying data to and from GPU memory.

`blocking` : optional
    Determines whether the external source should wait until data is available or just fail
    when the data is not available.

`no_copy` : optional
    Determines whether DALI should copy the buffer when feed_input is called.

    If set to True, DALI passes the user memory directly to the pipeline, instead of copying it.
    It is the user responsibility to keep the buffer alive and unmodified until it is
    consumed by the pipeline.

    The buffer can be modified or freed again after the output of the relevant iterations
    has been consumed. Effectively, it happens after ``prefetch_queue_depth`` or
    ``cpu_queue_depth * gpu_queue_depth`` (when they are not equal) iterations following
    the ``feed_input`` call.

    The memory location must match the specified ``device`` parameter of the operator.
    For the CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
    For the GPU, to avoid extra copy, the provided buffer must be contiguous. If you provide a list
    of separate Tensors, there will be an additional copy made internally, consuming both memory
    and bandwidth.
"""

    def __init__(self, source = None, num_outputs = None, *, cycle = None, layout = None, name = None, device = "cpu",
                 cuda_stream = None, use_copy_kernel = None, batch_size=None, **kwargs):
        self._schema = _b.GetSchema("_ExternalSource")
        self._spec = _b.OpSpec("_ExternalSource")
        self._device = device
        self._layout = layout
        self._cuda_stream = cuda_stream
        self._use_copy_kernel = use_copy_kernel
        self._batch_size = batch_size

        import nvidia.dali.ops
        kwargs, self._call_args = nvidia.dali.ops._separate_kwargs(kwargs)

        callback = _get_callback_from_source(source, cycle)

        if name is not None and num_outputs is not None:
            raise ValueError("`num_outputs` is not compatible with named `ExternalSource`")

        self._name = name
        self._num_outputs = num_outputs
        self._callback = callback

        self._spec.AddArg("device", device)
        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return False

    def __call__(self, *, source = None, cycle = None, name = None, layout = None, cuda_stream = None,
                 use_copy_kernel = None, **kwargs):
        ""
        from nvidia.dali.ops import _OperatorInstance

        if source is None:
            if cycle is not None:
                if self._callback:
                    raise ValueError("The argument ``cycle`` can only be specified if ``source`` is an iterable object "
                        "or a generator function specified in this call. To cycle through an iterable specified in "
                        "``__init__``, set ``cycle`` there.")
                else:
                    raise ValueError("The argument ``cycle`` can only be specified if ``source`` is a "
                                     "reusable iterable or a generator function.")
            callback = self._callback
        else:
            if self._callback is not None:
                raise RuntimeError("``source`` already specified in constructor.")
            callback = _get_callback_from_source(source, cycle)

        if self._layout is not None:
            if layout is not None:
                raise RuntimeError("``layout`` already specified in constructor.")
            else:
                layout = self._layout

        if self._cuda_stream is not None:
            if cuda_stream is not None:
                raise RuntimeError("``cuda_stream`` already specified in constructor.")
            else:
                cuda_stream = self._cuda_stream

        if self._use_copy_kernel is not None:
            if use_copy_kernel is not None:
                raise RuntimeError("``use_copy_kernel`` already specified in constructor.")
            else:
                use_copy_kernel = self._use_copy_kernel

        if name is None:
            name = self._name

        if name is not None and self._num_outputs is not None:
            raise RuntimeError("``num_outputs`` is not compatible with named ``ExternalSource``")

        if self._num_outputs is not None:
            outputs = []
            kwargs = {}
            group = _ExternalSourceGroup(callback, True, cuda_stream=cuda_stream, use_copy_kernel=use_copy_kernel, batch_size=self._batch_size)
            for i in range(self._num_outputs):
                op_instance = _OperatorInstance([], self, **kwargs)
                op_instance._callback = callback
                op_instance._output_index = i
                op_instance._group = group
                if layout is not None:
                    if isinstance(layout, (list, tuple)):
                        op_instance._layout = layout[i] if i < len(layout) else ""
                    else:
                        op_instance._layout = layout
                else:
                    op_instance._layout = None

                group.append(op_instance)
                op_instance.generate_outputs()
                outputs.append(op_instance.unwrapped_outputs)

            return outputs
        else:
            if name is not None:
                kwargs["name"] = name
            op_instance = _OperatorInstance([], self, **kwargs)
            op_instance._callback = callback
            op_instance._output_index = None
            op_instance._group = _ExternalSourceGroup(callback, False, [op_instance], cuda_stream=cuda_stream,
                                                      use_copy_kernel=use_copy_kernel)
            op_instance._layout = layout
            op_instance.generate_outputs()

            return op_instance.unwrapped_outputs

    __doc__ += _args_doc
    __call__.__doc__ += _args_doc


def _is_external_source_with_callback(op_instance):
    return isinstance(op_instance._op, ExternalSource) and op_instance._callback is not None

def external_source(source = None, num_outputs = None, *, cycle = None, name = None, device = "cpu", layout = None,
                    cuda_stream = None, use_copy_kernel = None, **kwargs):
    """Creates a data node which is populated with data from a Python source.
The data can be provided by the ``source`` function or iterable, or it can be provided by
``pipeline.feed_input(name, data, layout, cuda_stream)`` inside ``pipeline.iter_setup``.

In the case of the GPU input, it is the user responsibility to modify the
provided GPU memory content only using provided stream (DALI schedules a copy on it
and all work is properly queued). If no stream is provided feeding input blocks until the
provided memory is copied to the internal buffer.

.. warning::
    :meth:`nvidia.dali.ops.ExternalSource` operator is not compatible with TensorFlow integration.

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
    """

    if num_outputs is not None:
        if source is None:
            raise ValueError("The parameter ``num_outputs`` is only valid when using ``source`` to "
                "provide data. To feed multiple external sources in ``feed_input``, use multiple "
                "``external_source`` nodes.")

    op = ExternalSource(device = device, num_outputs = num_outputs, source = source,
                        cycle = cycle, layout = layout, cuda_stream = cuda_stream,
                        use_copy_kernel = use_copy_kernel, **kwargs)
    return op(name = name)

external_source.__doc__ += ExternalSource._args_doc
