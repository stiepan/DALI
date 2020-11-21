import multiprocessing
import numpy as np
from multiprocessing import shared_memory


def free_shared_mem(shm):
    shm.close()
    shm.unlink()


class MemBatchSerialized(object):

    def __init__(self, mem_batch_id, shm_name, size, samples):
        self.mem_batch_id = mem_batch_id
        self.shm_name = shm_name
        self.size = size
        self.samples = samples

    @classmethod
    def from_producer_batch(cls, batch):
        return cls(batch.mem_batch_id, batch.shm.name, batch.size, batch.samples)


def deserialize_sample(shm, sample):
    if isinstance(sample, NPSerialized):
        offset = sample.offset
        buffer = shm.buf[offset:offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if any(isinstance(sample, t) for t in (tuple, list,)):
        return type(sample)(deserialize_sample(shm, part) for part in sample)
    return sample


class MemBatchConsumerMgr(object):

    def __init__(self):
        self.batch_pool = {}

    def add_batch(self, batch):
        shm = self.batch_pool.get(batch.mem_batch_id)
        if shm is not None:
            if shm.name == batch.shm_name:
                return shm
            shm.close()
            del self.batch_pool[batch.mem_batch_id]
        try:
            shm = shared_memory.SharedMemory(name=batch.shm_name)
            self.batch_pool[batch.mem_batch_id] = shm
            return shm
        except:
            if shm is not None:
                shm.close()
            raise

    def deserialize_batch(self, shm, samples):
        # weakref.finalize(a, unlink_shm, shm)
        return [(idx, deserialize_sample(shm, sample)) for (idx, sample) in samples]

    def load_batch(self, batch):
        shm = self.add_batch(batch)
        return self.deserialize_batch(shm, batch.samples)


def arrays_size(sample):
    if isinstance(sample, tuple) or isinstance(sample, list):
        return sum(map(arrays_size, sample))
    if isinstance(sample, np.ndarray):
        return sample.nbytes
    return 0


class MemBatchProducer(object):

    def __init__(self, mem_batch_id, capacity):
        self.mem_batch_id = mem_batch_id
        self.samples = []
        self.size = 0
        self.shm = None
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=capacity)
        except:
            free_shared_mem(self.shm)
            self.shm = None
            raise

    def _resize(self, new_capacity=None, copy_data=True):
        capacity = self.capacity
        new_capacity = new_capacity or 2 * capacity
        assert(not copy_data or new_capacity >= capacity)
        new_shm = None
        # print("Resizing membatch {} {}".format(capacity, self.mem_batch_id))
        try:
            new_shm = shared_memory.SharedMemory(create=True, size=new_capacity)
            if copy_data:
                new_shm.buf[:capacity] = self.shm.buf
            else:
                self.reset_samples()
            free_shared_mem(self.shm)
            self.shm = new_shm
        except:
            if new_shm is not None:
                free_shared_mem(new_shm)
            raise

    def reset_samples(self):
        self.samples = []
        self.size = 0

    @property
    def capacity(self):
        return self.shm.size

    def free(self):
        print("Freeing shared mem {}".format(self.shm))
        if self.shm is not None:
            free_shared_mem(self.shm)
            self.shm = None

    def fill_batch(self, batch):
        if not batch:
            return
        needed_capacity = sum(map(arrays_size, [sample for _, sample in batch]))
        if self.capacity < needed_capacity:
            self._resize(max([needed_capacity, 2 * self.capacity]), False)
        for idx, sample in batch:
            self.add_sample_to_batch(idx, sample)

    def _add_array_to_batch(self, np_array):
        sample_size = np_array.nbytes
        if self.size + sample_size > self.capacity:
            self._resize()
        offset = self.size
        self.size += sample_size
        buffer = self.shm.buf[offset:offset + sample_size]
        shared_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=buffer)
        assert(shared_array.nbytes == sample_size)
        shared_array[:] = np_array[:]
        return NPSerialized.from_np(offset, shared_array)

    def _add_sample_to_batch(self, sample):
        if isinstance(sample, np.ndarray):
            return self._add_array_to_batch(sample)
        if any(isinstance(sample, t) for t in (tuple, list,)):
            return type(sample)(self._add_sample_to_batch(part) for part in sample)
        return sample

    def add_sample_to_batch(self, idx, sample):
        self.samples.append((idx, self._add_sample_to_batch(sample)))


class NPSerialized(object):

    def __init__(self, offset, shape, dtype, nbytes):
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.nbytes = nbytes

    @classmethod
    def from_np(cls, offset, np_array):
        return cls(offset, np_array.shape, np_array.dtype, np_array.nbytes)