import time
import heapq

min_slice = 1000

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
        self.time_before = time.monotonic_ns()
        self.tick_time = 0

    def update_min_vruntime(self):
        if(len(self.queue) != 0):
            if(self.running_task):
                self.min_vruntime = max(
                    self.min_vruntime,
                    min(self.running_task.vruntime, self.queue[0].vruntime)
                )
            else:
                self.min_vruntime = max(self.min_vruntime, self.queue[0].vruntime)

    def calculate_vruntime(self, task):
        now = time.monotonic_ns()
        curr_time = now
        delta_exec = (curr_time - task.exec_start) // 1000000
        delta_vruntime = (delta_exec * nice_weights[0]) // nice_weights[task.nice]
        task.vruntime += delta_vruntime
        task.exec_start = now
        

    def enqueue_task(self, task):
        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self.queue, task)

    def put_curr_task(self):
        if self.running_task:
            self.running_task.avg_burst_time += (time.monotonic_ns() - self.running_task.exec_start) / self.running_task.burst_count
            self.calculate_vruntime(self.running_task)
            heapq.heappush(self.queue, self.running_task)
            print(f"Task: {self.running_task.pid} finished running with avg burst time: {self.running_task.avg_burst_time}")
            self.running_task = None

    def set_task(self, task):
        if(self.running_task != None):
            self.put_curr_task()
        self.update_min_vruntime()
        self.running_task = task
        self.running_task.exec_start = time.monotonic_ns()
        self.running_task.burst_count += 1
        self.update_min_vruntime()
        self.tick_time = 0

    def tick(self):
        while True:
            now = time.monotonic_ns()
            delta_ns = now - self.time_before
            self.time_before = now
            self.tick_time += delta_ns // 1000000  

            if self.running_task:
                self.calculate_vruntime(self.running_task)
                if self.queue:
                    if (self.running_task.vruntime > self.queue[0].vruntime and self.tick_time >= min_slice):
                        self.set_task(heapq.heappop(self.queue))
            else:
                if self.queue:
                    self.set_task(heapq.heappop(self.queue))

            time.sleep(0.1)


class Task:
    def __init__(self, pid,nice):
        self.pid = pid
        self.vruntime = 0
        self.avg_burst_time = 0
        self.avg_wait_time = 0
        self.exec_start = 0
        self.nice = nice
        self.burst_count = 0

    def __lt__(self, other):
        return self.vruntime < other.vruntime


def main():
    sched = Scheduler()
    task1 = Task(1,0)
    task2 = Task(2,-3)
    task3 = Task(3,3)
    sched.enqueue_task(task1)
    sched.enqueue_task(task2)
    sched.enqueue_task(task3)
    sched.tick()


if __name__ == "__main__":
    main()

