import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import weakref
from nvidia.dali.memory_pool import SharedBatchWriter, SharedBatchMemCache, SharedChunkSerialized, SharedBatchBuffer


def arrays_size(sample):
    if isinstance(sample, tuple) or isinstance(sample, list):
        return sum(map(arrays_size, sample))
    if isinstance(sample, np.ndarray):
        return sample.nbytes
    return 0


def worker(worker_id, callback, task_queue, res_queue):
    batch_mem_cache = None
    try:
        batch_mem_cache = SharedBatchMemCache()
        print("Worker {} starts".format(worker_id))
        while True:
            idxs = task_queue.get()
            assert(isinstance(idxs, list))
            if (idxs == []):
                print("Worker {} finishing".format(worker_id))
                break
            batch_data = [(idx, callback(idx)) for idx in idxs]
            size_required = sum(map(arrays_size, (sample for _, sample in batch_data)))
            res_queue.put((worker_id, size_required))
            mem_chunk = task_queue.get()
            assert(isinstance(mem_chunk, SharedBatchBuffer))
            writer = SharedBatchWriter.from_shared_batch_buffer(batch_mem_cache, mem_chunk)
            writer.fill_with_batch(batch_data)
            res_queue.put(SharedChunkSerialized.from_writer(writer))
    except KeyboardInterrupt:
        print("KI {}".format(worker_id))
    finally:
        if batch_mem_cache is not None:
            batch_mem_cache.close_all_shm()


