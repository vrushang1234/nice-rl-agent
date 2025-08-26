import heapq
from rl_agent import RLAgent

min_slice = 50

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

BASE_SLICE = 3000

class Scheduler:
    def __init__(self):
        self.agent = RLAgent()
        self.running_task = None
        self.queue = []
        self.sleep_queue = []
        heapq.heapify(self.queue)
        self.min_vruntime = 0
        self.global_tick_time = 0

    def update_curr_se(self, task):
        now = self.global_tick_time
        delta_exec = now - task.exec_start
        task.exec_start = now
        task.sum_exec_runtime += delta_exec
        return delta_exec

    def calc_delta_fair(self, delta,task):
            delta_exec = (delta * nice_weights[0])//nice_weights[task.nice]
            return delta_exec

    def update_deadline(self, task):
        if task.vruntime - task.deadline < 0:
            return False
        task.deadline = task.vruntime + self.calc_delta_fair(BASE_SLICE, task)
        return True

    def update_curr(self,task):
        delta_exec = self.update_curr_se(task)
        task.vruntime += self.calc_delta_fair(delta_exec, task)
        task.resched = self.update_deadline(task) and bool(self.queue)
        self.update_min_vruntime()


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
        task.exec_start = now
        if delta_exec <=0:
            return
        delta_vruntime = (delta_exec * nice_weights[0]) // nice_weights[task.nice]
        task.vruntime += delta_vruntime
        self.update_min_vruntime()

    def enqueue_task(self, task):
        if task.deadline == 0:  # init request at first enqueue
            task.deadline = task.vruntime + self.calc_delta_fair(BASE_SLICE, task)
        task.wait_time_before = self.global_tick_time
        task.wait_time_count += 1
        (action, nice) = self.agent.policy_decide([task.avg_burst_time, task.avg_wait_time])
        task.nice = nice
        task.action = action
        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self.queue, task)

    def __enqueue_task(self, task):
        (action,nice) = self.agent.policy_decide([task.avg_burst_time, task.avg_wait_time])
        task.nice = nice
        task.action = action

        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self.queue, task)
        task.wait_time_before = self.global_tick_time
        task.wait_time_count += 1

    def check_sleep(self):
        i=0
        while i < len(self.sleep_queue):
            curr = self.sleep_queue[i]
            curr.sleep_left-=1
            if curr.sleep_left==0:
                self.__enqueue_task(curr)
                self.sleep_queue.pop(i)
            else:
                i+=1

    def put_curr_task(self):
        if self.running_task:
            self.running_task.resched = False
            burst_length = self.global_tick_time - self.running_task.burst_start_time
            self.running_task.total_burst_time += burst_length
            self.running_task.avg_burst_time = self.running_task.total_burst_time / self.running_task.burst_count
            self.agent.calculate_reward([self.running_task.avg_burst_time, self.running_task.avg_wait_time], self.running_task.action)
            if self.running_task.sleep_time > 0:
                self.running_task.sleep_left = self.running_task.sleep_time
                self.sleep_queue.append(self.running_task)
            else:
                self.__enqueue_task(self.running_task)
            self.running_task = None

    def set_task(self, task):
        if self.running_task is not None:
            self.put_curr_task()
        task.total_wait_time += self.global_tick_time - task.wait_time_before
        task.avg_wait_time = task.total_wait_time / task.wait_time_count
        self.running_task = task
        self.running_task.exec_start = self.global_tick_time
        self.running_task.burst_start_time = self.global_tick_time
        self.running_task.burst_count += 1

    def tick(self):
        while True:
            self.global_tick_time += 1
            self.check_sleep()
            if self.running_task:
                self.update_curr(self.running_task)
                burst_time = self.global_tick_time - self.running_task.burst_start_time
                if((self.running_task.resched and burst_time > min_slice) or (burst_time >= self.running_task.max_burst_time)  or (self.queue[0].deadline < self.running_task.deadline and burst_time > min_slice)):
                        self.set_task(heapq.heappop(self.queue))
            else:
                if self.queue:
                    self.set_task(heapq.heappop(self.queue))


class Task:
    def __init__(self, pid, nice, max_burst_time, sleep_time):
        self.pid = pid
        self.vruntime = 0
        self.burst_start_time = 0
        self.deadline = 0
        self.max_burst_time = max_burst_time
        self.avg_burst_time = 0
        self.burst_count = 0
        self.total_burst_time = 0
        self.sum_exec_runtime = 0

        self.avg_wait_time = 0
        self.wait_time_before = 0
        self.total_wait_time = 0
        self.wait_time_count = 0

        self.sleep_time = sleep_time
        self.sleep_left = 0

        self.exec_start = 0
        self.nice = nice
        self.action = 0

    def __lt__(self, other):
        return self.deadline < other.deadline


def main():
    sched = Scheduler()
    task1 = Task(1, 0, 30, 10)
    task2 = Task(2, 0, 10, 5)
    task3 = Task(3, 0, 20000000, 0)
    sched.enqueue_task(task1)
    sched.enqueue_task(task2)
    sched.enqueue_task(task3)
    sched.tick()


if __name__ == "__main__":
    main()

