import multiprocessing
from nvidia.dali.worker import worker
from nvidia.dali.memory_pool import MemBatchConsumerMgr


class WorkersPool(object):

    def __init__(self, callback, workers_no=None):
        method = "fork"
        print("context for method {}".format(method))
        mp = multiprocessing.get_context(method)
        self.mem_pool_mgr = MemBatchConsumerMgr()
        # self.round_counter = 0
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

    # def next_worker(self):
    #     counter = self.round_counter
    #     self.round_counter = (self.round_counter + 1) % self.workers_no
    #     return counter

    def process_batch(self, tasks):
        tasks = list(tasks)
        if not tasks:
            return []
        tasks_no = len(tasks)
        chunk_size = tasks_no // self.workers_no
        queued_no = chunk_size + (tasks_no % self.workers_no)
        self.task_queues[0].put(tasks[:queued_no])
        queued_workers = 1
        if chunk_size:
            for worker_id in range(1, self.workers_no):
                self.task_queues[worker_id].put(tasks[queued_no:queued_no + chunk_size])
                queued_no += chunk_size
            queued_workers = self.workers_no
        done_tasks = {}
        for _ in range(queued_workers):
            worker_batch_serialized = self.res_queue.get()
            worker_batch = self.mem_pool_mgr.load_batch(worker_batch_serialized)
            done_tasks.update(worker_batch)
        return [done_tasks[task_id] for task_id in tasks]
        # return [(np.frombuffer(buf, dt, np.product(shape)).reshape(shape), np.frombuffer(buf1, dt1, np.product(shape1)).reshape(shape1)) for (buf, shape, dt), (buf1, shape1, dt1) in buffers]