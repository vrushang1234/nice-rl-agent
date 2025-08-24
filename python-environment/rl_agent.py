import numpy as np
from nn_policy import NeuralNet

class RLAgent:
    def __init__(self):
        self.nn = NeuralNet()
    
    def policy_decide(self, state):
        temp_state = np.array(state, dtype=np.float32).reshape(-1,1)
        (action,nice) = self.nn.select_action(temp_state)
        return action,nice

    def calculate_reward(self, state, action):
        burst,wait = 0.9, -0.9
        reward = state[0] * burst + state[1] * wait
        temp_state = np.array(state, dtype=np.float32).reshape(-1,1)
        self.nn.update(temp_state, action, reward)