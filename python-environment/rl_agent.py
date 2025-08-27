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
