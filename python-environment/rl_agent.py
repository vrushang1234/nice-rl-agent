import numpy as np
from policy_nn import PolicyNeuralNet
from value_function_nn import ValueFunction
from policy_test import PolicyNeuralNetTest

class RLAgent:
    def __init__(self, test=False):
        self.policy = PolicyNeuralNet() if test==False else PolicyNeuralNetTest()
        self.value_function = ValueFunction()

    def rl_policy_decide(self,state):
        temp_state = np.array(state)
        output = self.policy.forward(temp_state)
        return output

    def train_for_ten_epochs(self, env_params, s_T):
        states = [t[0] for t in env_params]
        actions = [t[2] for t in env_params]
        old_ps = [t[3] for t in env_params]
        self.value_function.set_v_old(states + [s_T])
        for i in range(10):
            r_t = self.calculate_rt(states, actions, old_ps)
            _, g = self.value_function.update(env_params, r_t)
            self.policy.policy_backprop_step(env_params, g)

    def calculate_rt(self, states, actions, old_probs):
        eps = 1e-8
        r_t = []
        for s, a, p_old in zip(states, actions, old_probs):
            p_new = float(self.rl_policy_decide(s)[a])
            if not np.isfinite(p_old) or p_old <= 0.0:
                p_old = eps
            r_t.append(p_new / p_old)
        return np.asarray(r_t, dtype=np.float64)
    
    def calculate_reward(self, state):
        wait_time = -0.4
        burst_time = 0.1
        avg_wait_time_diff = -0.25
        avg_burst_time_diff = +0.25
        reward = state[0] * wait_time + state[1] * burst_time + (state[2] - state[0]) * avg_wait_time_diff + (state[3] - state[1]) * avg_burst_time_diff
        return reward
