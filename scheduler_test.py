import copy
import numpy as np
from scheduler import Scheduler, Task, WorkloadGenerator

def run_ab_with_generated_batch(n=15, seed=1234, runs=10):
    rl_results  = []
    cfs_results = []

    for i in range(runs):
        np.random.seed(seed + i)  # vary seed each run

        rl_sched  = Scheduler(rl_enabled=True,  test=True)
        cfs_sched = Scheduler(rl_enabled=False, test=False)

        gen = WorkloadGenerator(seed=seed + i, arrival_prob=1.0)
        batch = gen.generate_batch(n)

        for t in batch:
            rl_sched.enqueue_task(copy.deepcopy(t))
        for t in batch:
            cfs_sched.enqueue_task(copy.deepcopy(t))

        # run RL scheduler
        while True:
            if rl_sched.tick(workload_gen=None):
                break
        rl_avg = rl_sched.turnaround_sum / rl_sched.finished_count
        rl_results.append(rl_avg)

        # run CFS scheduler
        while True:
            if cfs_sched.tick(workload_gen=None):
                break
        cfs_avg = cfs_sched.turnaround_sum / cfs_sched.finished_count
        cfs_results.append(cfs_avg)

        print(f"Run {i+1}: RL Avg Turnaround = {rl_avg:.3f} ticks, "
              f"CFS Avg Turnaround = {cfs_avg:.3f} ticks")

    # overall averages
    print("\n==== Overall Results ====")
    print(f"RL  Mean Avg Turnaround: {np.mean(rl_results):.3f} ticks")
    print(f"CFS Mean Avg Turnaround: {np.mean(cfs_results):.3f} ticks")

if __name__ == "__main__":
    run_ab_with_generated_batch(n=15, seed=3421, runs=10)

