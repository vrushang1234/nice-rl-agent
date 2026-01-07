import heapq
import numpy as np
import matplotlib.pyplot as plt

from rl_agent import RLAgent

nice_actions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

slice_actions = [
    7000, 17500, 28000, 38500, 49000,
    59500, 70000, 80500, 91000, 101500, 112000
]

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

BASE_SLICE = 70000
MAX_TRAIN_COUNT = 100


def plot_rewards(rewards, window=200):
    rewards = np.array(rewards)
    if len(rewards) < window:
        smooth = rewards
    else:
        smooth = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )

    plt.figure()
    plt.plot(smooth)
    plt.xlabel("Training Step")
    plt.ylabel("Smoothed Reward")
    plt.title("RL Scheduler Training Reward")
    plt.grid(True)
    plt.show()


def rolling_mean(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


class Scheduler:
    def __init__(self, rl_enabled=True, test=False):
        self.running_task = None
        self.queue = []
        self.sleep_queue = []
        heapq.heapify(self.queue)

        self.min_vruntime = 0
        self.global_tick_time = 0

        self.avg_wait_time = 0
        self.avg_burst_time = 0

        self.agent = RLAgent(test) if rl_enabled else None

        self.last_state = []
        self.last_action = -1
        self.last_action_prob = None

        self.steps = []
        self.train_count = 0

        self.turnaround_sum = 0
        self.finished_count = 0

        self.rl_enabled = rl_enabled
        self.test = test

        self.ctx_switches = 0

        self.wait_time_log = []
        self.burst_time_log = []
        self.reward_log = []
        self.training_complete = False

        self.episode_reward_sum = 0.0
        self.episode_reward_count = 0

    def record_turnaround(self, task):
        time = task.total_wait_time + task.sum_exec_runtime
        self.turnaround_sum += float(time)
        self.finished_count += 1

    def update_curr_se(self, task):
        now = self.global_tick_time
        delta_exec = now - task.exec_start
        task.exec_start = now
        task.sum_exec_runtime += delta_exec
        return delta_exec

    def calc_delta_fair(self, delta, task):
        return (delta * nice_weights[0]) // nice_weights[task.nice]

    def update_deadline(self, task):
        if task.vruntime - task.deadline < 0:
            return False
        task.deadline = task.vruntime + self.calc_delta_fair(task.slice, task)
        return True

    def update_curr(self, task):
        delta_exec = self.update_curr_se(task)
        task.vruntime += self.calc_delta_fair(delta_exec, task)

        task.total_runtime -= delta_exec
        if task.total_runtime <= 0:
            task.finished = True

        task.resched = self.update_deadline(task) and bool(self.queue)
        self.update_min_vruntime()

    def update_min_vruntime(self):
        if self.queue:
            if self.running_task:
                self.min_vruntime = max(
                    self.min_vruntime,
                    min(self.running_task.vruntime, self.queue[0].vruntime)
                )
            else:
                self.min_vruntime = max(
                    self.min_vruntime,
                    self.queue[0].vruntime
                )

    def calculate_vruntime(self, task):
        now = self.global_tick_time
        delta_exec = now - task.exec_start
        task.exec_start = now

        if delta_exec <= 0:
            return

        delta_vruntime = (
            delta_exec * nice_weights[0]
        ) // nice_weights[task.nice]

        task.vruntime += delta_vruntime
        self.update_min_vruntime()

    def enqueue_task(self, task):
        if self.last_state == [] and self.last_action == -1:
            self.last_state = [
                task.last_wait_time,
                task.avg_wait_time,
                task.last_burst_time,
                task.avg_burst_time,
                self.avg_wait_time,
                self.avg_burst_time,
            ]

        if task.deadline == 0:
            task.deadline = task.vruntime + self.calc_delta_fair(BASE_SLICE, task)

        task.wait_time_before = self.global_tick_time
        task.wait_time_count += 1
        task.vruntime = max(task.vruntime, self.min_vruntime)

        heapq.heappush(self.queue, task)

    def __enqueue_task(self, task):
        total_w = 0.0
        total_b = 0.0
        n = 0

        for q in self.queue:
            total_w += q.avg_wait_time
            total_b += q.avg_burst_time
            n += 1

        for q in self.sleep_queue:
            total_w += q.avg_wait_time
            total_b += q.avg_burst_time
            n += 1

        if self.running_task:
            total_w += self.running_task.avg_wait_time
            total_b += self.running_task.avg_burst_time
            n += 1

        if n > 0:
            self.avg_wait_time = total_w / n
            self.avg_burst_time = total_b / n

        if self.last_state == [] and self.last_action == -1:
            self.last_state = [
                task.last_wait_time,
                task.avg_wait_time,
                task.last_burst_time,
                task.avg_burst_time,
                self.avg_wait_time,
                self.avg_burst_time,
            ]

        if self.rl_enabled:
            probs = self.agent.rl_policy_decide(self.last_state)

            if not np.all(np.isfinite(probs)):
                probs = np.ones_like(probs) / len(probs)

            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / s

            action = int(np.random.choice(len(probs), p=probs))
            self.last_action = action
            self.last_action_prob = float(probs[action])

            task.slice = slice_actions[action]
            task.deadline = task.vruntime + self.calc_delta_fair(task.slice, task)

        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self.queue, task)

        task.wait_time_before = self.global_tick_time
        task.wait_time_count += 1

    def check_sleep(self):
        i = 0
        while i < len(self.sleep_queue):
            curr = self.sleep_queue[i]
            curr.sleep_left -= 1

            if curr.sleep_left <= 0:
                self.__enqueue_task(curr)
                self.sleep_queue.pop(i)
            else:
                i += 1

    def put_curr_task(self):
        if not self.running_task:
            return

        if self.rl_enabled:
            reward = self.agent.calculate_reward([
                self.running_task.avg_wait_time,
                self.running_task.avg_burst_time,
                self.avg_wait_time,
                self.avg_burst_time,
                self.ctx_switches,
            ])


            self.episode_reward_sum += reward
            self.episode_reward_count += 1

            if (
                self.last_state and
                self.last_action != -1 and
                self.last_action_prob is not None
            ):
                self.steps.append(
                    (self.last_state, reward, self.last_action, self.last_action_prob)
                )

            if len(self.steps) >= 8000:
                if self.episode_reward_count > 0:
                    mean_reward = (
                        self.episode_reward_sum /
                        self.episode_reward_count
                    )
                    self.reward_log.append(mean_reward)

                self.episode_reward_sum = 0.0
                self.episode_reward_count = 0

                if self.queue:
                    task = self.queue[0]
                elif self.sleep_queue:
                    task = self.sleep_queue[0]
                else:
                    task = self.running_task

                self.last_state = [
                    task.last_wait_time,
                    task.avg_wait_time,
                    task.last_burst_time,
                    task.avg_burst_time,
                    self.avg_wait_time,
                    self.avg_burst_time,
                ]

                self.agent.train_for_ten_epochs(self.steps, self.last_state)

                self.steps = []
                self.train_count += 1
                print("Train Count:", self.train_count)

                if self.train_count >= MAX_TRAIN_COUNT:
                    print("Training Complete")
                    self.training_complete = True
                    return

            self.last_state = []
            self.last_action = -1
            self.last_action_prob = None

        self.running_task.resched = False

        burst_length = self.global_tick_time - self.running_task.burst_start_time
        self.burst_time_log.append(burst_length)

        self.running_task.total_burst_time += burst_length
        self.running_task.last_burst_time = burst_length
        self.running_task.avg_burst_time = (
            self.running_task.total_burst_time /
            max(1, self.running_task.burst_count)
        )

        if self.running_task.finished:
            self.record_turnaround(self.running_task)
            self.running_task = None
            return

        if self.running_task.sleep_time > 0:
            self.running_task.sleep_left = self.running_task.sleep_time
            self.sleep_queue.append(self.running_task)
        else:
            self.__enqueue_task(self.running_task)

        self.running_task = None

    def set_task(self, task):
        if self.running_task is not None:
            self.put_curr_task()

        wait = self.global_tick_time - task.wait_time_before
        task.total_wait_time += wait
        task.last_wait_time = wait
        task.avg_wait_time = (
            task.total_wait_time /
            max(1, task.wait_time_count)
        )

        self.wait_time_log.append(wait)

        self.running_task = task
        self.running_task.exec_start = self.global_tick_time
        self.running_task.burst_start_time = self.global_tick_time
        self.running_task.burst_count += 1

    def tick(self, workload_gen=None):
        while True:
            if self.training_complete:
                return False

            if not self.running_task and not self.queue and not self.sleep_queue:
                return True

            self.global_tick_time += 1

            if workload_gen:
                new_task = workload_gen.maybe_spawn(self)
                if new_task:
                    self.enqueue_task(new_task)

            self.check_sleep()

            if self.running_task:
                self.update_curr(self.running_task)

                burst_time = (
                    self.global_tick_time -
                    self.running_task.burst_start_time
                )

                should_preempt = (
                    self.running_task.resched or
                    burst_time >= self.running_task.max_burst_time or
                    (
                        self.queue and
                        self.queue[0].deadline < self.running_task.deadline
                    ) or
                    self.running_task.finished
                )

                if should_preempt:
                    if self.queue:
                        self.set_task(heapq.heappop(self.queue))
                        self.ctx_switches += 1
                    else:
                        self.put_curr_task()
            else:
                if self.queue:
                    self.set_task(heapq.heappop(self.queue))


class Task:
    def __init__(self, pid, nice, max_burst_time, sleep_time, total_runtime):
        self.pid = pid
        self.vruntime = 0
        self.deadline = 0
        self.sum_exec_runtime = 0

        self.burst_start_time = 0
        self.max_burst_time = max_burst_time
        self.avg_burst_time = 0
        self.burst_count = 0
        self.total_burst_time = 0
        self.last_burst_time = 0

        self.avg_wait_time = 0
        self.wait_time_before = 0
        self.total_wait_time = 0
        self.wait_time_count = 0
        self.last_wait_time = 0

        self.sleep_time = sleep_time
        self.sleep_left = 0

        self.exec_start = 0
        self.nice = nice
        self.action = 0
        self.slice = BASE_SLICE

        self.total_runtime = int(total_runtime)
        self.finished = False

    def __lt__(self, other):
        return self.deadline < other.deadline


class WorkloadGenerator:
    def __init__(
        self,
        seed=42,
        arrival_prob=0.02,
        cpu_bound_share=0.35,
        io_bound_share=0.35,
        bursty_share=0.20,
        background_share=0.10,
        max_active=40,
        max_total=None,
        target_active=16,
    ):
        self.rng = np.random.RandomState(seed)
        self.arrival_prob = float(arrival_prob)
        self.profile_cdf = np.cumsum([
            cpu_bound_share,
            io_bound_share,
            bursty_share,
            background_share,
        ])
        self.next_pid = 100

        self.max_active = int(max_active)
        self.max_total = None if max_total is None else int(max_total)
        self.spawned_total = 0
        self.target_active = int(target_active)

    def maybe_spawn(self, sched):
        active = (
            (1 if sched.running_task else 0) +
            len(sched.queue) +
            len(sched.sleep_queue)
        )

        if active >= self.max_active:
            return None

        if self.max_total is not None and self.spawned_total >= self.max_total:
            return None

        scale = max(0.1, self.target_active / max(active, 1))
        p = self.arrival_prob * scale

        if self.rng.rand() > p:
            return None

        u = self.rng.rand()
        if u < self.profile_cdf[0]:
            task = self._cpu_bound()
        elif u < self.profile_cdf[1]:
            task = self._io_bound()
        elif u < self.profile_cdf[2]:
            task = self._bursty()
        else:
            task = self._background()

        self.spawned_total += 1
        return task

    def _cpu_bound(self):
        pid = self._pid()
        max_burst = int(self.rng.randint(2333, 9334))
        sleep = 0
        total_runtime = int(self.rng.randint(46666, 186667))
        return Task(pid, 0, max_burst, sleep, total_runtime)

    def _io_bound(self):
        pid = self._pid()
        max_burst = int(self.rng.randint(116, 701))
        sleep = int(self.rng.randint(467, 4667))
        total_runtime = int(self.rng.randint(11667, 70001))
        return Task(pid, 0, max_burst, sleep, total_runtime)

    def _bursty(self):
        pid = self._pid()
        if self.rng.rand() < 0.5:
            max_burst = int(self.rng.randint(116, 584))
            sleep = int(self.rng.randint(1167, 7001))
        else:
            max_burst = int(self.rng.randint(1867, 4667))
            sleep = int(self.rng.randint(0, 701))
        total_runtime = int(self.rng.randint(23334, 116667))
        return Task(pid, 0, max_burst, sleep, total_runtime)

    def _background(self):
        pid = self._pid()
        max_burst = int(self.rng.randint(4667, 46667))
        sleep = int(self.rng.randint(0, 234))
        total_runtime = int(self.rng.randint(233334, 1166667))
        return Task(pid, 0, max_burst, sleep, total_runtime)

    def _pid(self):
        self.next_pid += 1
        return self.next_pid

    def generate_batch(self, n):
        tasks = []
        for _ in range(int(n)):
            u = self.rng.rand()
            if u < self.profile_cdf[0]:
                t = self._cpu_bound()
            elif u < self.profile_cdf[1]:
                t = self._io_bound()
            elif u < self.profile_cdf[2]:
                t = self._bursty()
            else:
                t = self._background()
            tasks.append(t)
            self.spawned_total += 1
        return tasks


def main():
    sched = Scheduler()

    task1 = Task(1, 0, 700, 234, total_runtime=46667)
    task2 = Task(2, 0, 234, 117, total_runtime=23334)
    task3 = Task(3, 0, 4666667, 0, total_runtime=116667)

    sched.enqueue_task(task1)
    sched.enqueue_task(task2)
    sched.enqueue_task(task3)

    gen = WorkloadGenerator(
        seed=999,
        arrival_prob=0.02,
        max_active=20,
        max_total=None,
        target_active=12,
    )

    while True:
        empty = sched.tick(workload_gen=gen)
        if empty or sched.training_complete:
            break

    plot_rewards(sched.reward_log)
    plt.plot(rolling_mean(sched.reward_log, 20))


if __name__ == "__main__":
    main()

