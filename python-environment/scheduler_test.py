import copy
import numpy as np
from scheduler import Scheduler, Task, WorkloadGenerator

def run_ab_with_generated_batch(n=15, seed=1234):
    np.random.seed(seed)

    rl_sched  = Scheduler(rl_enabled=True,  test=True)   # or False, as you prefer
    cfs_sched = Scheduler(rl_enabled=False, test=False)

    gen = WorkloadGenerator(seed=seed, arrival_prob=1.0)  # arrival_prob irrelevant for batch
    batch = gen.generate_batch(n)

    for t in batch:
        rl_sched.enqueue_task(copy.deepcopy(t))
    for t in batch:
        cfs_sched.enqueue_task(copy.deepcopy(t))

    while True:
        if rl_sched.tick(workload_gen=None):
            break
    print(f"RL Avg Turnaround: {rl_sched.turnaround_sum/rl_sched.finished_count:.3f} ticks")

    while True:
        if cfs_sched.tick(workload_gen=None):
            break
    print(f"CFS Avg Turnaround: {cfs_sched.turnaround_sum/cfs_sched.finished_count:.3f} ticks")

if __name__ == "__main__":
    run_ab_with_generated_batch(n=15, seed=1234)

