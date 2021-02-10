# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import socket
import threading
import multiprocessing
from collections import OrderedDict
from nvidia.dali import backend as _b
from nvidia.dali.worker import worker
from nvidia.dali.messages import ScheduledTasks
from nvidia.dali.shared_batch import SharedBatchMeta
from nvidia.dali.shared_batch import deserialize_sample_meta, deserialize_batch, import_numpy
from nvidia.dali import shared_mem


class SharedBatchesConsumer:
    """Counterpart of worker.py:SharedBatchesDispatcher. Can receive and deserialize batch
    from the worker, keeps track of already received memory chunks and opens new chunks
    or resizes exiting ones if necessary."""
    class MemChunk:
        def __init__(self, shm_chunk: shared_mem.SharedMem, capacity: int):
            self.shm_chunk = shm_chunk
            self.capacity = capacity

    def __init__(self):
        import_numpy()
        self.batch_pool = {}

    def get_mem_chunk(self, sock: socket.socket, batch: SharedBatchMeta) -> MemChunk:
        """Get the fd for shared memory through sock and mmap the memory based on metadata
        in `batch`. Adjust if it was previously mapped.

        Parameters
        ----------
        `sock` : socket.socket
            Socket used to transfer file descriptors.
        `batch` : SharedBatchMeta
            Serialized batch metadata

        Returns
        -------
        MemChunk
            Wrapped SharedMem object with current capacity containing the obtained data.
        """
        chunk = self.batch_pool.get(batch.mem_chunk_id)
        if chunk is not None:
            if chunk.capacity != batch.capacity:
                chunk.shm_chunk.resize(batch.capacity, trunc=False)
                chunk.capacity = batch.capacity
            return chunk
        fd, shm_chunk = -1, None
        try:
            fd = multiprocessing.reduction.recv_handle(sock)
            assert os.fstat(fd).st_size >= batch.capacity
            shm_chunk = shared_mem.SharedMem.open(fd, batch.capacity)
        except:
            if shm_chunk is not None:
                shm_chunk.close()
            # close fd manually if shm_chunk creation failed, otherwise shm_chunk
            # is responsible for doing so
            elif fd >= 0:
                os.close(fd)
            raise
        chunk = self.MemChunk(shm_chunk, batch.capacity)
        self.batch_pool[batch.mem_chunk_id] = chunk
        return chunk

    def load_batch(self, sock: socket.socket, batch_meta: SharedBatchMeta):
        """Based on the metadata in `batch` obtain the smem mapping, obtain the sample metadata
        and deserialize the resulting data.

        Returns
        -------
        List of (sample id: int, sample)
            Indexed samples from the obtained part of batch
        """
        chunk = self.get_mem_chunk(sock, batch_meta)
        # Load list of indexed SampleMeta: (idx, SampleMeta) or (idx, tuple of SampleMeta)
        sample_meta = deserialize_sample_meta(chunk.shm_chunk, batch_meta)
        return deserialize_batch(chunk.shm_chunk, sample_meta)


class CallbackContext:
    """Counterpart of worker.py:CallbackContext, for every callback that can be run
in the workers stores shared memory chunks that are used to passed callback results, keeps track
of what tasks has been scheduled to run in the workers and what partial results has been
received so far.
"""

    def __init__(self):
        self.batch_consumer = SharedBatchesConsumer()
        self.reset()

    def reset(self):
        self.partially_received = {}
        self.scheduled = OrderedDict()
        self.iter_failed = {}

    def push_scheduled(self, batch_i, tasks):
        """Mark batch of given id as scheduled with all passed tasks. Initialize structures
        for receiving the results.
        """
        if batch_i in self.scheduled:
            raise RuntimeError("Given batch has already been scheduled")
        self.partially_received[batch_i] = {}
        self.scheduled[batch_i] = tasks

    def pop_scheduled(self):
        return self.scheduled.popitem(last=False)

    def handle_error(self, batch_i):
        """Check if given batch notified error and raise it"""
        if batch_i in self.iter_failed:
            exception, traceback_str = self.iter_failed[batch_i]
            del self.iter_failed[batch_i]
            del self.partially_received[batch_i]
            if isinstance(exception, StopIteration):
                raise exception
            else:
                # Raise new exception propagating the traceback from worker thread as error
                # message, originating from original exception
                raise Exception(
                    "\n\nException traceback received from worker thread:\n\n" + traceback_str) from exception

    def is_error(self, batch_i):
        return batch_i in self.iter_failed

    def set_error(self, batch_i, excpetion, traceback_str):
        self.iter_failed[batch_i] = (excpetion, traceback_str)

    def is_cleared(self, batch_i):
        return batch_i not in self.partially_received

    def is_not_received(self, batch_i, tasks):
        """Check if we didn't receive all results for given tasks and batch_i"""
        return len(self.partially_received[batch_i]) < len(tasks)

    def get_batch(self, batch_i, tasks):
        """Return the full batch, mark it as cleared and consumed"""
        full_batch = self.partially_received[batch_i]
        res = [full_batch[i] for i, _ in tasks]
        del full_batch
        return res

    def receive_chunk(self, batch_i, sock, batch_serialized):
        """Obtain the chunk and decode it, add to partially gathered result"""
        worker_batch = self.batch_consumer.load_batch(sock, batch_serialized)
        self.partially_received[batch_i].update(worker_batch)


class ProcPool:
    """Runs pool of worker processes, stores pipes and sockets used to communicate with the workers,
starts thread keeping track of running processes and initializes communication.
"""

    def __init__(
            self, callbacks, prefetch_queue_depths, workers_num=None, init_method="fork",
            initial_chunk_size=1024 * 1024):
        if len(callbacks) != len(prefetch_queue_depths):
            raise RuntimeError("Number of prefetch queues must match number of callbacks")
        if any(prefetch_queue_depth <= 0 for prefetch_queue_depth in prefetch_queue_depths):
            raise RuntimeError("Prefetch queue must have at least one element")
        if initial_chunk_size <= 0:
            raise RuntimeError("Chunk capacity must be positive integer")
        if init_method == 'fork' and _b.HasCudaContext():
            raise RuntimeError(
                "Cannot fork process when there is CUDA context bound to it. "
                "Make sure you build pipeline before CUDA context is acquired or change Python workers "
                "starting method from ``fork`` to ``spawn`` (see Piepline's ``py_workers_init`` option). "
                "If you are trying to build multiple pipelines that use Python workers, you will need to "
                "call ``start_py_workers`` method on all of them before calling ``build`` method of any pipeline.")
        mp = multiprocessing.get_context(init_method)
        if worker is None:
            workers_num = mp.cpu_count()
        if workers_num < 1:
            raise RuntimeError("workers_num must be a positive integer")
        self._workers_num = workers_num
        self._processes = []
        self._task_pipes = []
        self._res_pipes = []
        self._socks = []
        self._task_pipes_lock = threading.Lock()
        self._from_tracker = None
        self._to_tracker = None
        self._tracker_thread = None
        for i in range(self._workers_num):
            task_r, task_w = mp.Pipe(duplex=False)
            res_r, res_w = mp.Pipe(duplex=False)
            sock_reader, sock_writer = socket.socketpair()
            process = mp.Process(
                target=worker,
                args=(i, callbacks, prefetch_queue_depths, initial_chunk_size,
                      task_r, res_w, sock_writer),
            )
            self._task_pipes.append(task_w)
            self._res_pipes.append(res_r)
            self._processes.append(process)
            self._socks.append(sock_reader)
        self._start_processes()

    def get_recv_pipes(self):
        """Return all pipes with incoming communication.

        Note: One pipe is from tracking process that may send `None` in case of shutdown.
        """
        return self._res_pipes + [self._from_tracker]

    @property
    def workers_num(self):
        return self._workers_num

    @property
    def task_pipes_lock(self):
        return self._task_pipes_lock

    def sock(self, worker_id: int):
        return self._socks[worker_id]

    def send(self, worker_id: int, scheduled_tasks: ScheduledTasks):
        """Send a message scheduling a task to worker `worker_id`. Needs to be done while
        holding the `ProcPool.task_pipes_lock`."""
        self._task_pipes[worker_id].send(scheduled_tasks)

    def close(self):
        if self._tracker_thread is None:
            return
        try:
            self._to_tracker.send(None)
        except BrokenPipeError:
            """workers already exitied, tracker_thread finished its task and exitied closing the pipe"""
        self._tracker_thread.join()
        self._tracker_thread = None

    def _start_processes(self):
        try:
            for process in self._processes:
                process.start()
            from_tracker_r, from_tracker_w = multiprocessing.Pipe(duplex=False)
            to_tracker_r, to_tracker_w = multiprocessing.Pipe(duplex=False)
            self._from_tracker = from_tracker_r
            self._to_tracker = to_tracker_w
            # thread is properly joined in .close method, but it is run as daemon to prevent Python
            # from trying to join it automatically too early on cleanup
            self._tracker_thread = threading.Thread(
                target=join_thread, args=(
                    self._processes, to_tracker_r, from_tracker_w, self._task_pipes, self._task_pipes_lock
                ), daemon=True)
            self._tracker_thread.start()
        except:
            for proc in self._processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self._processes:
                if proc.pid is not None:
                    proc.join()
            raise


def join_thread(processes, tracker_pipe, main_thread_pipe, task_pipes, task_pipes_lock):
    """Observer thread for ProcPool used for joining processes and distributing
    stop signal (`None` message).

    Parameters
    ----------
    `processes` : List of multiprocessing.Process
        Worker processes.
    `tracker_pipe` : Pipe
        Read pipe for communicating stop to join_thread from main process.
    `main_thread_pipe` : Pipe
        Pipe where stop will be sent for main thread.
    `task_pipes` : List of Pipe
        Pipes where tasks are sent to worker processes, used to signal stop.
    `task_pipes_lock`
        Lock for accessing task pipes.
    """
    ps = {p.sentinel: p for p in processes}
    listen_for = list(ps.keys()) + [tracker_pipe]
    try:
        # Once one process exits stop the whole group (gracefully if possible)
        while True:
            sentinels = multiprocessing.connection.wait(listen_for)
            proc_sentinels = [s for s in sentinels if s != tracker_pipe]
            if tracker_pipe in sentinels or any(
                    ps[sentinel].exitcode is not None for sentinel in proc_sentinels):
                break
        with task_pipes_lock:
            for proc, task_pipe in zip(processes, task_pipes):
                if proc.exitcode is None:
                    task_pipe.send(None)
    except:
        for proc in processes:
            if proc.exitcode is not None:
                proc.terminate()
        raise
    finally:
        for proc in processes:
            proc.join(1)
        for proc in processes:
            if proc.exitcode is None:
                proc.terminate()
                proc.join()
        # if workers exited unexpectedly main thread might be blocked waiting for the data;
        # let it know it is over
        main_thread_pipe.send(None)


class WorkersPool:
    """"Combines worker processes pool with callback contexts, can be used to schedule batches
    to be run on the workers and to receive resulting batches from the workers."""

    def __init__(self, callbacks_num, pool):
        """
        Parameters
        ----------
        `callbacks_num` : int
            Number of callabacks that can be run in the workers, each callback will have separate
            context with dedicated shared memory pool.
        `pool` : ProcPool
            ProcPool instance enabling basic communication with worker processes.
        """
        self.contexts = [CallbackContext() for _ in range(callbacks_num)]
        self.pool = pool
        self.rec_pipes = self.pool.get_recv_pipes()

    @classmethod
    def from_groups(
            cls, groups, keep_alive_queue_size, init_method="fork", workers_num=None,
            initial_chunk_size=1024 * 1024):
        """Creates new WorkerPool instance for given list of ExternalSource groups.

        Parameters
        ----------
        `groups` : _ExternalSourceGroup list
            List of external source groups.
        `keep_alive_queue_size` : int
            Number of the most recently produced batches whose underlying shared memory should
            remain untouched (because they might still be referenced further in the pipeline).
            Note that actual number of simultaneously kept batches will probably be greater
            because of additional prefetching.
        `init_method` : str
            Method of starting worker processes, either fork or spawn.
        `workers_num` : int
            Number of workers to be created in ProcPool.
        `initial_chunk_size` : int
            Initial size of each shared memory chunk.
        """
        callbacks = [group.callback for group in groups]
        queue_depths = [keep_alive_queue_size + group.prefetch_queue_depth for group in groups]
        pool = ProcPool(callbacks, queue_depths, workers_num, init_method, initial_chunk_size)
        return cls(len(callbacks), pool)

    def schedule_batch(self, context_i, batch_i, tasks):
        """Distribute `tasks` among workers to run them by calling `context_i`th callaback

        Parameters
        ----------
        `context_i` : int
            Specifies which callback will be used to run the task, it must be the index corresponding
            to the order of callbacks passed when constructing WorkerPool.
        `batch_i` : int
            Ordinal of the batch that tasks list corresponds to.
        `tasks` : nvidia.dali.types.SampleInfo list
            You can think of resulting batch as [callback(task) for task in tasks] with the exception that
            callbacks will be run in parallel.
        """
        tasks = list(enumerate(tasks))
        if not tasks:
            raise RuntimeError("Cannot schedule empty list of tasks")
        context = self.contexts[context_i]
        if context.iter_failed:
            # there is no point in scheduling anything for the context that has reached the end of data
            # or failed with error, once user receives batch that raised exception they should reset
            # the context before scheduling new tasks
            return
        self._distribute(context_i, batch_i, tasks)
        # TODO check if raising from doubly scheduled task makes sense?
        context.push_scheduled(batch_i, tasks)

    def _distribute(self, context_i, batch_i, tasks):
        workers_num = self.pool.workers_num
        tasks_no = len(tasks)
        chunk_size = tasks_no // workers_num
        remainder = tasks_no % workers_num
        queued_no = 0
        with self.pool.task_pipes_lock:
            for worker_id in range(workers_num):
                worker_chunk = chunk_size + (worker_id < remainder)
                if worker_chunk == 0:
                    break
                scheduled_tasks = ScheduledTasks(
                    context_i, batch_i, tasks[queued_no: queued_no + worker_chunk])
                queued_no += worker_chunk
                self.pool.send(worker_id, scheduled_tasks)

    def receive_batch(self, context_i):
        """Returns the next produced batch (in the order of schedule_batch calls) for the
        ``context_i``th callaback.

        Parameters
        ----------
        `context_i` : int
            Specifies which callback you want the results from, ordering corresponds to the order of
            callbacks passed when constructing the pool.
        """
        context = self.contexts[context_i]
        assert len(context.scheduled) > 0, "No task has been scheduled"
        batch_i, tasks = context.pop_scheduled()
        while context.is_not_received(batch_i, tasks) and not context.is_error(batch_i):
            self._receive_chunk()
        context.handle_error(batch_i)
        res = context.get_batch(batch_i, tasks)
        return res

    def _receive_chunk(self):
        ready_workers = multiprocessing.connection.wait(self.rec_pipes)
        for worker_pipe in ready_workers:
            completed_tasks = worker_pipe.recv()
            if completed_tasks is None:
                raise RuntimeError("Worker exited unexpectedly")
            worker_id = completed_tasks.worker_id
            batch_i = completed_tasks.batch_i
            if completed_tasks.context_i == -1:
                assert completed_tasks.is_failed(), "Anonymous task can only be failed"
                # Raise new exception propagating the traceback from worker thread as error
                # message, originating from original exception. There is no originating context
                # or batch, so we just reraise the error.
                raise Exception(
                    "\n\nException traceback received from worker thread:\n\n" + \
                     completed_tasks.traceback_str) from completed_tasks.exception
            context = self.contexts[completed_tasks.context_i]
            # batch has been discarded
            if context.is_cleared(batch_i) or context.is_error(batch_i):
                continue
            # iteration failed with exception
            if completed_tasks.is_failed():
                context.set_error(batch_i, completed_tasks.exception, completed_tasks.traceback_str)
            # received a valid chunk
            else:
                context.receive_chunk(
                    batch_i, self.pool.sock(worker_id),
                    completed_tasks.batch_serialized)

    def reset(self):
        for context in self.contexts:
            context.reset()

    def reset_context(self, context_i):
        self.contexts[context_i].reset()

    def close(self):
        self.pool.close()
