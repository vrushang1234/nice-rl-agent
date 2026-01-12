# Python Simulator for Adaptive RL Scheduler

This repository contains a Python-based simulator used to design, train, and evaluate an adaptive reinforcement learning (RL)-based CPU scheduler before integration into the Linux kernel.

The simulator provides a controlled environment for experimenting with scheduling policies, reward functions, and workload characteristics without kernel-level complexity.

---

## Overview

Developing and debugging scheduling policies directly inside the Linux kernel is slow and error-prone. This simulator acts as a rapid prototyping and evaluation platform for learning-based schedulers.

It models key scheduling concepts such as:
- Task arrival and execution
- CPU burst behavior
- Wait time accumulation
- Context switching

The simulator enables iterative policy development and offline training prior to deployment in the kernel.

---

## Key Features

- Discrete-event CPU scheduling simulator
- Reinforcement learning agent interface
- PPO-based policy training
- Configurable workloads and task distributions
- Metrics collection for wait time, burst time, and fairness
- Visualization utilities for performance comparison

---

## Simulator Design

### Tasks
Each task is modeled with:
- Arrival time
- CPU burst length
- Priority / nice value
- Execution state

### Scheduler Loop
At each scheduling decision point:
1. The simulator constructs a compact state representation
2. The RL agent selects an action
3. Scheduling parameter (slice) is updated
4. Metrics are recorded for evaluation

---

## Reinforcement Learning Setup

- Algorithm: Proximal Policy Optimization (PPO)
- Policy network: small MLP
- Training is performed entirely in Python
- Learned weights are exported for fixed-point inference in the kernel

The simulator mirrors kernel constraints where possible to ensure policy transferability.

---

## Metrics & Evaluation

The simulator tracks:
- Average wait time
- Average burst time
- Turnaround time
- Context switch frequency

Results can be compared against baseline schedulers such as CFS-like heuristics to evaluate tradeoffs and stability.

---

## Running the Simulator 

### Training the agent
```python simulator.py```

### Evaluate the trained agent
```python test.py```
