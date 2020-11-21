import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import weakref
from nvidia.dali.memory_pool import MemBatchProducer, MemBatchSerialized


def worker(proc_id, callback, task_queue, res_queue):
    batches = None
    try:
        depth = 3
        capacity = 1 * 1024 * 1024
        batches = [MemBatchProducer("batch_{}_{}".format(proc_id, i), capacity) for i in range(depth)]
        batch_i = depth
        print("Worker {} starts".format(proc_id))
        while True:
            batch_i = (batch_i + 1) % depth
            batch_mem = batches[batch_i]
            batch_mem.reset_samples()
            idxs = task_queue.get()
            assert(isinstance(idxs, list))
            if (idxs == []):
                break
            batch_data = [(idx, callback(idx)) for idx in idxs]
            batch_mem.fill_batch(batch_data)
            res_queue.put(MemBatchSerialized.from_producer_batch(batch_mem))
    except KeyboardInterrupt:
        print("KI {}".format(proc_id))
    finally:
        if batches is not None:
            for batch in batches:
                batch.free()


