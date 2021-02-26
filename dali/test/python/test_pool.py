# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pool import WorkerPool, ProcPool
from nvidia.dali.types import SampleInfo
import numpy as np
import os
import time

def answer(pid, info):
    return np.array([pid, info.idx_in_epoch, info.idx_in_batch, info.iteration])

def simple_callback(info):
    pid = os.getpid()
    return answer(pid, info)

def create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork"):
    queue_depths = [queue_depth for _ in callbacks]
    proc_pool = ProcPool(callbacks, queue_depths, num_workers=num_workers,
                         start_method=start_method, initial_chunk_size=1024 * 1024)
    worker_pool = WorkerPool(len(callbacks), proc_pool)
    return worker_pool

def get_internal_pids(worker_pool):
    return [proc.pid for proc in worker_pool.pool._processes]


# ################################################################################################ #
# 1 callback, 1 worker tests
# ################################################################################################ #


def test_pool_one_task():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks = [(SampleInfo(0, 0, 0),)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    batch = pool.receive_batch(context_i=0)
    for task, sample in zip(tasks, batch):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


def test_pool_multi_task():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks = [(SampleInfo(i, i, 0),) for i in range(10)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    batch = pool.receive_batch(context_i=0)
    for task, sample in zip(tasks, batch):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


# Even though we receive 1 batch, it already should be overwritten by the result
# of calculating the second batch, just in case we wait a few seconds
def test_pool_overwrite_single_batch():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks_0 = [(SampleInfo(0, 0, 0),)]
    tasks_1 = [(SampleInfo(1, 0, 1),)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks_0)
    pool.schedule_batch(context_i=0, batch_i=1, tasks=tasks_1)
    time.sleep(5)
    batch_0 = pool.receive_batch(context_i=0)
    batch_1 = pool.receive_batch(context_i=0)
    for task, sample in zip(tasks_1, batch_0):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    for task, sample in zip(tasks_1, batch_1):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


# Test that with bigger queue depth we will still overwrite the memory used as the results
def test_pool_overwrite_multiple_batch():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=3, num_workers=1, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks_list = [(i, [(SampleInfo(i, 0, i),)]) for i in range(4)]
    for i, tasks in tasks_list:
        pool.schedule_batch(context_i=0, batch_i=i, tasks=tasks)
    batches = [pool.receive_batch(context_i=0) for i in range(4)]
    tasks_batches = zip(tasks_list, batches)
    _, tasks_3 = tasks_list[3]
    for (i, tasks), batch in tasks_batches:
        if i == 0:
            tasks_to_compare = tasks_3
        else:
            tasks_to_compare = tasks
        for task, sample in zip(tasks_to_compare, batch):
            np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


# Test that we can hold as many results as the queue depth
def test_pool_no_overwrite_batch():
    callbacks = [simple_callback]
    for depth in [1, 2, 4, 8]:
        pool = create_pool(callbacks, queue_depth=depth, num_workers=1, start_method="fork")
        pids = get_internal_pids(pool)
        pid = pids[0]
        tasks_list = [(i, [(SampleInfo(i, 0, i),)]) for i in range(depth)]
        for i, tasks in tasks_list:
            pool.schedule_batch(context_i=0, batch_i=i, tasks=tasks)
        batches = [pool.receive_batch(context_i=0) for i in range(depth)]
        tasks_batches = zip(tasks_list, batches)
        for (i, tasks), batch in tasks_batches:
            for task, sample in zip(tasks, batch):
                np.testing.assert_array_equal(answer(pid, *task), sample)
        pool.close()


# ################################################################################################ #
# 1 callback, multiple workers tests
# ################################################################################################ #


def test_pool_work_split_2_tasks():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=2, start_method="fork")
    pids = get_internal_pids(pool)
    tasks = [(SampleInfo(0, 0, 0),), (SampleInfo(1, 1, 0),)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    batch = pool.receive_batch(context_i=0)
    for task, sample, pid in zip(tasks, batch, pids):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


def test_pool_work_split_multiple_tasks():
    callbacks = [simple_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=2, start_method="fork")
    num_tasks = 16
    pids = get_internal_pids(pool)
    tasks = [(SampleInfo(i, i, 0),) for i in range(num_tasks)]
    split_pids = []
    assert num_tasks % len(pids) == 0, "Testing only even splits"
    for pid in pids:
        split_pids += [pid] * (num_tasks // len(pids))
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    batch = pool.receive_batch(context_i=0)
    for task, sample, pid in zip(tasks, batch, split_pids):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    pool.close()


# ################################################################################################ #
# multiple callback, 1 worker tests
# ################################################################################################ #


def another_callback(info):
    return simple_callback(info) + 100

def test_pool_many_ctxs():
    callbacks = [simple_callback, another_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks = [(SampleInfo(0, 0, 0),)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    pool.schedule_batch(context_i=1, batch_i=0, tasks=tasks)
    batch_0 = pool.receive_batch(context_i=0)
    batch_1 = pool.receive_batch(context_i=1)
    for task, sample, pid in zip(tasks, batch_0, pids):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    for task, sample, pid in zip(tasks, batch_1, pids):
        np.testing.assert_array_equal(answer(pid, *task) + 100, sample)
    pool.close()


# Check that the same worker executes the ctxs
def test_pool_many_ctxs_many_workers():
    callbacks = [simple_callback, another_callback]
    pool = create_pool(callbacks, queue_depth=1, num_workers=5, start_method="fork")
    pids = get_internal_pids(pool)
    pid = pids[0]
    tasks = [(SampleInfo(0, 0, 0),)]
    pool.schedule_batch(context_i=0, batch_i=0, tasks=tasks)
    pool.schedule_batch(context_i=1, batch_i=0, tasks=tasks)
    batch_0 = pool.receive_batch(context_i=0)
    batch_1 = pool.receive_batch(context_i=1)
    for task, sample, pid in zip(tasks, batch_0, pids):
        np.testing.assert_array_equal(answer(pid, *task), sample)
    for task, sample, pid in zip(tasks, batch_1, pids):
        np.testing.assert_array_equal(answer(pid, *task) + 100, sample)
    pool.close()



# TODO(klecki): Add the cleanup checkup, and test "spawn" as well
