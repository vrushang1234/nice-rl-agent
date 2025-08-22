import time
import heapq

min_slice = 100

nice_weights = {
    -5: 3121,
    -4: 2607,
    -3: 2171,
    -2: 1820,
    -1: 1529,
     0: 1024,
     1: 820,
     2: 655,
     3: 526,
     4: 423,
     5: 335,
}

class Scheduler:
    def __init__(self):
        self.running_task = None
        self.queue = []
        heapq.heapify(self.queue)
        self.min_vruntime = 0
        self.global_tick_time = 0 

    def update_min_vruntime(self):
        if len(self.queue) != 0:
            if self.running_task:
                self.min_vruntime = max(
                    self.min_vruntime,
                    min(self.running_task.vruntime, self.queue[0].vruntime)
                )
            else:
                self.min_vruntime = max(self.min_vruntime, self.queue[0].vruntime)

    def calculate_vruntime(self, task):
        now = self.global_tick_time
        delta_exec = now - task.exec_start
        delta_vruntime = (delta_exec * nice_weights[0]) // nice_weights[task.nice]
        task.vruntime += delta_vruntime

    def enqueue_task(self, task):
        task.wait_time_before = self.global_tick_time
        task.wait_time_count += 1
        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self.queue, task)

    def put_curr_task(self):
        if self.running_task:
            burst_length = self.global_tick_time - self.running_task.exec_start
            self.running_task.total_burst_time += burst_length
            self.running_task.avg_burst_time = self.running_task.total_burst_time / self.running_task.burst_count
            self.calculate_vruntime(self.running_task)
            heapq.heappush(self.queue, self.running_task)
            print(f"Task: {self.running_task.pid} finished burst {burst_length} | Avg burst time: {self.running_task.avg_burst_time} | Avg wait time: {self.running_task.avg_wait_time}")
            self.running_task = None

    def set_task(self, task):
        if self.running_task is not None:
            self.put_curr_task()
        task.total_wait_time += self.global_tick_time - task.wait_time_before
        task.avg_wait_time = task.total_wait_time / task.wait_time_count
        self.update_min_vruntime()
        self.running_task = task
        self.running_task.exec_start = self.global_tick_time
        self.running_task.burst_count += 1
        self.update_min_vruntime()

    def tick(self):
        while True:
            self.global_tick_time += 1
            if self.running_task:
                self.calculate_vruntime(self.running_task)
                if self.queue:
                    burst_length = self.global_tick_time - self.running_task.exec_start
                    if ((self.running_task.vruntime > self.queue[0].vruntime and burst_length >= min_slice)
                        or (burst_length >= self.running_task.max_burst_time)):
                        self.set_task(heapq.heappop(self.queue))
            else:
                if self.queue:
                    self.set_task(heapq.heappop(self.queue))
            time.sleep(0.01)


class Task:
    def __init__(self, pid, nice, max_burst_time):
        self.pid = pid
        self.vruntime = 0
        self.max_burst_time = max_burst_time
        self.avg_burst_time = 0
        self.burst_count = 0
        self.total_burst_time = 0

        self.avg_wait_time = 0
        self.wait_time_before = 0
        self.total_wait_time = 0
        self.wait_time_count = 0

        self.exec_start = 0
        self.nice = nice

    def __lt__(self, other):
        return self.vruntime < other.vruntime


def main():
    sched = Scheduler()
    task1 = Task(1, 0, 50)
    task2 = Task(2, -3, 200)
    task3 = Task(3, 3, 100)
    sched.enqueue_task(task1)
    sched.enqueue_task(task2)
    sched.enqueue_task(task3)
    sched.tick()


if __name__ == "__main__":
    main()

