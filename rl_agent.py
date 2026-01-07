import numpy as np
from policy_nn import PolicyNeuralNet
from policy_test import PolicyNeuralNetTest
from value_function_nn import ValueFunction


class RLAgent:
    def __init__(self, test=False):
        self.policy = PolicyNeuralNet() if not test else PolicyNeuralNetTest()
        self.value_function = ValueFunction()

        self.prev_avg_wait = None
        self.prev_avg_burst = None
        self.prev_ctx_switches = None

    def rl_policy_decide(self, state):
        return self.policy.forward(np.asarray(state))

    def calculate_rt(self, states, actions, old_probs):
        eps = 1e-8
        ratios = []

        for s, a, p_old in zip(states, actions, old_probs):
            p_new = float(self.rl_policy_decide(s)[a])
            p_old = max(p_old, eps)
            ratios.append(p_new / p_old)

        return np.asarray(ratios, dtype=np.float64)

    def calculate_reward(self, state):
        avg_wait, avg_burst, ctx = state[2], state[3], state[4]

        if self.prev_avg_wait is None:
            self.prev_avg_wait = avg_wait
            self.prev_avg_burst = avg_burst
            self.prev_ctx_switches = ctx
            return 0.0

        dw = avg_wait - self.prev_avg_wait
        db = self.prev_avg_burst - avg_burst
        dc = self.prev_ctx_switches - ctx

        self.prev_avg_wait = avg_wait
        self.prev_avg_burst = avg_burst
        self.prev_ctx_switches = ctx

        return -dw + 0.5 * db + 0.1 * dc

    def train_for_ten_epochs(self, env_params, s_T):
        states = [t[0] for t in env_params]
        rewards = [t[1] for t in env_params]
        actions = [t[2] for t in env_params]
        old_probs = [t[3] for t in env_params]

        self.value_function.set_v_old(states + [s_T])

        advantages, returns = self.value_function.compute_gae(rewards)

        for _ in range(10):
            r_t = self.calculate_rt(states, actions, old_probs)
            self.policy.policy_backprop_step(env_params, advantages, r_t)
            self.value_function.update(states, returns)

