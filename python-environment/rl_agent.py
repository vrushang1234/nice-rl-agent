import numpy as np
from policy_nn import PolicyNeuralNet
from value_function_nn import ValueFunction

class RLAgent:
    def __init__(self):
        self.policy = PolicyNeuralNet()
        self.value_function = ValueFunction()
        self.old_pi = []

    def rl_policy_decide(self,state):
        temp_state = np.array(state)
        output = self.policy.forward(temp_state)
        return output

    def train_for_fifty_epochs(self,env_params,s_T):
        states = [state[0] for state in env_params]
        self.value_function.set_v_old(states + [s_T])
        for _ in range(50):
            r_t = self.calculate_rt(states)
            _,g = self.value_function.update(env_params, r_t)
            self.policy.policy_backprop_step(env_params, g)
        self.old_pi = []

    def calculate_rt(self, states):
        r_t = [1] * len(states)
        eps = 1e-8
        if self.old_pi == []:
            for state in states:
                output = self.rl_policy_decide(state)
                self.old_pi.append((np.argmax(output), max(output)))
            return r_t
        for i in range(len(states)):
            output = self.rl_policy_decide(states[i])
            r_t[i] = np.exp(np.log(output[self.old_pi[i][0]] + eps) - np.log(self.old_pi[i][1] + eps))
        return r_t
    
    def calculate_reward(self, state):
        wait_time = -0.4
        burst_time = 0.1
        avg_wait_time_diff = -0.25
        avg_burst_time_diff = -0.25
        reward = state[0] * wait_time + state[1] * burst_time + (state[2] - state[0]) * avg_wait_time_diff + (state[3] - state[1]) * avg_burst_time_diff 
        return reward
