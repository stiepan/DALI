import random
import string
import multiprocessing
from multiprocessing import shared_memory
import numpy as np


def free_shared_mem(shm):
    shm.close()
    shm.unlink()


def deserialize_sample(buffer, sample):
    if isinstance(sample, NPSerialized):
        offset = sample.offset
        buffer = buffer[offset:offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if any(isinstance(sample, t) for t in (tuple, list,)):
        return type(sample)(deserialize_sample(buffer, part) for part in sample)
    return sample


class SharedBatchMem(object):

    SHARED_MEM_NAME_PREFIX = 'nvidia_dali'

    def __init__(self, mem_batch_id, capacity):
        self.mem_batch_id = mem_batch_id
        self.size = 0
        self.shm = None
        try:
            self.shm = self._new_shm_mem(capacity)
        except:
            if self.shm is not None:
                free_shared_mem(self.shm)
            self.shm = None
            raise

    def _create_name(self):
        random_suffix = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        return '_'.join([self.SHARED_MEM_NAME_PREFIX, str(self.mem_batch_id), random_suffix])

    def _new_shm_mem(self, capacity):
        return shared_memory.SharedMemory(self._create_name(), create=True, size=capacity)

    @property
    def capacity(self):
        return self.shm.size

    def resize(self, new_capacity=None, copy_data=True):
        capacity = self.capacity
        new_capacity = new_capacity or 2 * capacity
        assert(not copy_data or new_capacity >= capacity)
        new_shm = None
        print("Resizing membatch {} {} {}".format(capacity, new_capacity, self.mem_batch_id))
        try:
            new_shm = self._new_shm_mem(new_capacity)
            if copy_data:
                new_shm.buf[:capacity] = self.shm.buf
            else:
                self.clear()
            free_shared_mem(self.shm)
            self.shm = new_shm
        except:
            if new_shm is not None:
                free_shared_mem(new_shm)
            raise

    def assure_room_for(self, size):
        cur_capacity = self.capacity
        if cur_capacity < size:
            self.resize(max([size, 2 * cur_capacity]), False)

    def clear(self):
        self.size = 0

    def reserve(self, size):
        offset = self.size
        self.size += size
        return offset

    def free(self):
        print("Freeing shared mem {}".format(self.shm))
        if self.shm is not None:
            free_shared_mem(self.shm)
            self.shm = None


class SharedBatchPool(object):

    BATCH_MEM_CLASS = SharedBatchMem

    def __init__(self, batches_no, initial_capacity):
        assert(batches_no > 0 and initial_capacity > 0)
        self.round_counter = batches_no - 1
        self.count = batches_no
        self.mems = [self.BATCH_MEM_CLASS(i, initial_capacity) for i in range(batches_no)]

    def _next_round(self):
        self.round_counter = (self.round_counter + 1) % self.count
        return self.round_counter

    def next_batch(self):
        mem_batch = self.mems[self._next_round()]
        mem_batch.clear()
        return mem_batch


class SharedBatchBuffer(object):

    @classmethod
    def from_shared_batch_mem(cls, mem_batch, size):
        offset = mem_batch.reserve(size)
        return cls(offset, size, mem_batch.mem_batch_id, mem_batch.shm.name)

    def __init__(self, offset, size, mem_batch_id, shm_name):
        self.offset = offset
        self.size = size
        self.mem_batch_id = mem_batch_id
        self.shm_name = shm_name


class SharedBatchMemCache(object):

    def __init__(self):
        self.shm_pool = {}

    def _get_shm(self, batch_buffer):
        shm = self.shm_pool.get(batch_buffer.mem_batch_id)
        if shm is not None:
            # print('Got mem of name {} with batch buffer {} {} {}'.format(shm.name, batch_buffer.size, batch_buffer.offset, shm.buf.nbytes))
            if shm.name == batch_buffer.shm_name:
                return shm
            # print("Removing shm from cache {}".format(shm.name))
            shm.close()
            del self.shm_pool[batch_buffer.mem_batch_id]
        try:
            shm = shared_memory.SharedMemory(name=batch_buffer.shm_name)
            self.shm_pool[batch_buffer.mem_batch_id] = shm
            return shm
        except:
            if shm is not None:
                shm.close()
            raise

    def get_memview_buffer(self, batch_buffer):
        shm = self._get_shm(batch_buffer)
        return shm.buf[batch_buffer.offset:batch_buffer.offset + batch_buffer.size]

    def close_all_shm(self):
        for shm in self.shm_pool.values():
            shm.close()


class SharedBatchWriter(object):

    @classmethod
    def from_shared_batch_buffer(cls, mem_cache, batch_buffer):
        memview_buffer = mem_cache.get_memview_buffer(batch_buffer)
        return cls(batch_buffer.mem_batch_id, memview_buffer)

    def __init__(self, mem_batch_id, memview_buffer):
        self.mem_batch_id = mem_batch_id
        self.buffer = memview_buffer
        self.samples = []
        self.size = 0

    def _add_array_to_batch(self, np_array):
        sample_size = np_array.nbytes
        offset = self.size
        self.size += sample_size
        buffer = self.buffer[offset:(offset + sample_size)]
        # it actually may fail because [] operator for buffer won't let you access unlocated addresses
        assert(buffer.nbytes == sample_size)
        # if buffer.nbytes != sample_size:
            # print("ddd {} {} {} {}".format(buffer.nbytes, sample_size, self.buffer.nbytes, self.size))
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

    def fill_with_batch(self, batch):
        if not batch:
            return
        for idx, sample in batch:
            self.add_sample_to_batch(idx, sample)


class SharedChunkSerialized(object):

    @classmethod
    def from_writer(cls, batch_writer):
        return cls(batch_writer.mem_batch_id, batch_writer.samples)

    def __init__(self, mem_batch_id, samples):
        self.mem_batch_id = mem_batch_id
        self.samples = samples


class SharedBatchConsumer(object):

    def __init__(self, batch_pool):
        self.batch_pool = batch_pool

    def deserialize_batch(self, batch: SharedChunkSerialized):
        # weakref.finalize(a, unlink_shm, shm)
        buffer = self.batch_pool.mems[batch.mem_batch_id].shm.buf
        return [(idx, deserialize_sample(buffer, sample)) for (idx, sample) in batch.samples]


class NPSerialized(object):

    def __init__(self, offset, shape, dtype, nbytes):
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.nbytes = nbytes

    @classmethod
    def from_np(cls, offset, np_array):
        return cls(offset, np_array.shape, np_array.dtype, np_array.nbytes)