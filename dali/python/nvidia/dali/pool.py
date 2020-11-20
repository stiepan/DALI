import multiprocessing
from nvidia.dali.worker import worker, deserialize

class WorkersPool(object):

    def __init__(self, callback, workers_no=None):
        method = "fork"
        print("context for method {}".format(method))
        mp = multiprocessing.get_context(method)
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
        return [deserialize(done_tasks[task_id]) for task_id, _ in split_tasks]
        # return [(np.frombuffer(buf, dt, np.product(shape)).reshape(shape), np.frombuffer(buf1, dt1, np.product(shape1)).reshape(shape1)) for (buf, shape, dt), (buf1, shape1, dt1) in buffers]