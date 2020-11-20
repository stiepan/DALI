import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import weakref


class SharedNP(object):

    def __init__(self, shm_name, dtype, shape):
        self.shm_name = shm_name
        self.dtype = dtype
        self.shape = shape


def serialize_np(a):
    try:
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        b[:] = a[:]
        return shm, SharedNP(shm.name, a.dtype, a.shape)
    except:
        shm.close()
        shm.unlink()
        raise


def _serialize(res, shms):
    try:
        if isinstance(res, np.ndarray):
            if np.product(res.shape) < 100:
                return res
            shm, serialized = serialize_np(res)
            shms.append(shm)
            return serialized
        if isinstance(res, list) or isinstance(res, tuple):
            return type(res)(_serialize(r, shms) for r in res)
        assert(False)
    except:
        for shm in shms:
            shm.close()
            shm.unlink()
        shms.clear()


def serialize(res):
    shms = []
    return shms, _serialize(res, shms)


def unlink_shm(shm):
    # print("Freeing memory {}".format(shm.name))
    shm.close()
    shm.unlink()


def deserialize(obj):
    if isinstance(obj, SharedNP):
        shm = shared_memory.SharedMemory(name=obj.shm_name)
        a = np.ndarray(obj.shape, dtype=obj.dtype, buffer=shm.buf)
        weakref.finalize(a, unlink_shm, shm)
        return a
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)(deserialize(r) for r in obj)
    return obj


def worker(proc_id, callback, task_queue, res_queue):
    try:
        print("Worker {} starts".format(proc_id))
        while True:
            idx = task_queue.get()
            if (idx < 0):
                break
            res = callback(idx)
            _, serialized = serialize(res)
            res_queue.put((idx, serialized))
    except KeyboardInterrupt:
        print("KI {}".format(proc_id))

