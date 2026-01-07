import copy
import numpy as np
import matplotlib.pyplot as plt

from scheduler import Scheduler, Task, WorkloadGenerator


def moving_average(data, window=50):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_wait_comparison(rl_wait, cfs_wait, window=50):
    plt.figure()
    plt.plot(moving_average(rl_wait, window), label="RL Scheduler")
    plt.plot(moving_average(cfs_wait, window), label="CFS Scheduler")
    plt.xlabel("Scheduling Event Index")
    plt.ylabel("Wait Time (ticks)")
    plt.title("RL vs CFS – Smoothed Wait Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_turnaround_comparison(rl_turnaround, cfs_turnaround):
    plt.figure()
    plt.plot(rl_turnaround, marker="o", label="RL Scheduler")
    plt.plot(cfs_turnaround, marker="o", label="CFS Scheduler")
    plt.xlabel("Run Index")
    plt.ylabel("Average Turnaround Time (ticks)")
    plt.title("RL vs CFS – Average Turnaround Time per Run")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_ab_test(num_tasks=15, seed=124, runs=10):
    rl_turnaround, cfs_turnaround = [], []
    rl_wait_all, rl_burst_all = [], []
    cfs_wait_all, cfs_burst_all = [], []

    for run_id in range(runs):
        np.random.seed(seed + run_id)

        rl_sched = Scheduler(rl_enabled=True, test=True)
        cfs_sched = Scheduler(rl_enabled=False, test=False)

        generator = WorkloadGenerator(seed=seed + run_id, arrival_prob=1.0)
        task_batch = generator.generate_batch(num_tasks)

        for task in task_batch:
            rl_sched.enqueue_task(copy.deepcopy(task))
            cfs_sched.enqueue_task(copy.deepcopy(task))

        while not rl_sched.tick(workload_gen=None):
            pass

        rl_avg_turnaround = (
            rl_sched.turnaround_sum / rl_sched.finished_count
        )
        rl_avg_wait = np.mean(rl_sched.wait_time_log)

        rl_turnaround.append(rl_avg_turnaround)
        rl_wait_all.extend(rl_sched.wait_time_log)
        rl_burst_all.extend(rl_sched.burst_time_log)

        while not cfs_sched.tick(workload_gen=None):
            pass

        cfs_avg_turnaround = (
            cfs_sched.turnaround_sum / cfs_sched.finished_count
        )
        cfs_avg_wait = np.mean(cfs_sched.wait_time_log)

        cfs_turnaround.append(cfs_avg_turnaround)
        cfs_wait_all.extend(cfs_sched.wait_time_log)
        cfs_burst_all.extend(cfs_sched.burst_time_log)

        print(
            f"Run {run_id + 1}: "
            f"RL  Avg Turnaround = {rl_avg_turnaround:.3f}, "
            f"CFS Avg Turnaround = {cfs_avg_turnaround:.3f}, "
            f"RL  Avg Wait = {rl_avg_wait:.3f}, "
            f"CFS Avg Wait = {cfs_avg_wait:.3f}"
        )

    print("\n========== Overall Results ==========")
    print(f"RL  Mean Turnaround: {np.mean(rl_turnaround):.3f} ticks")
    print(f"CFS Mean Turnaround: {np.mean(cfs_turnaround):.3f} ticks")
    print(f"RL  Mean Wait Time: {np.mean(rl_wait_all):.3f} ticks")
    print(f"CFS Mean Wait Time: {np.mean(cfs_wait_all):.3f} ticks")

    plot_wait_comparison(rl_wait_all, cfs_wait_all, 500)
    plot_turnaround_comparison(rl_turnaround, cfs_turnaround)



if __name__ == "__main__":
    run_ab_test(num_tasks=15, seed=3421, runs=10)

