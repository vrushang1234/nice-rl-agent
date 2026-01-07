import numpy as np


class ValueFunction:
    def __init__(self):
        self.INPUT_SIZE = 6
        self.HIDDEN_1_SIZE = 50 
        self.HIDDEN_2_SIZE = 70 
        self.LR = 3e-4
        self.GAMMA = 0.9
        self.LAMBDA = 0.95

        self.w1 = np.random.randn(self.HIDDEN_1_SIZE, self.INPUT_SIZE) * np.sqrt(2 / self.INPUT_SIZE)
        self.b1 = np.zeros(self.HIDDEN_1_SIZE)

        self.w2 = np.random.randn(self.HIDDEN_2_SIZE, self.HIDDEN_1_SIZE) * np.sqrt(2 / self.HIDDEN_1_SIZE)
        self.b2 = np.zeros(self.HIDDEN_2_SIZE)

        self.w3 = np.random.randn(1, self.HIDDEN_2_SIZE) * np.sqrt(2 / self.HIDDEN_2_SIZE)
        self.b3 = np.zeros(1)

        self.v_old = None

    def forward(self, s):
        z1 = self.w1 @ s + self.b1
        a1 = np.tanh(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = np.tanh(z2)
        return (self.w3 @ a2 + self.b3).item()

    def set_v_old(self, states):
        self.v_old = np.array([self.forward(s) for s in states])

    def compute_gae(self, rewards):
        T = len(rewards)
        deltas = np.zeros(T)
        advantages = np.zeros(T)

        for t in range(T):
            deltas[t] = rewards[t] + self.GAMMA * self.v_old[t + 1] - self.v_old[t]

        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.GAMMA * self.LAMBDA * gae
            advantages[t] = gae

        returns = advantages + self.v_old[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, returns):
        T = len(states)

        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        gw3 = np.zeros_like(self.w3)
        gb3 = np.zeros_like(self.b3)

        for s, R in zip(states, returns):
            z1 = self.w1 @ s + self.b1
            a1 = np.tanh(z1)

            z2 = self.w2 @ a1 + self.b2
            a2 = np.tanh(z2)

            v = (self.w3 @ a2 + self.b3).item()

            dL_dv = (v - R) * 2.0

            gw3 += dL_dv * a2[None, :]
            gb3 += dL_dv

            da2 = dL_dv * self.w3[0]
            dz2 = da2 * (1.0 - a2 ** 2)

            gw2 += np.outer(dz2, a1)
            gb2 += dz2

            da1 = self.w2.T @ dz2
            dz1 = da1 * (1.0 - a1 ** 2)

            gw1 += np.outer(dz1, s)
            gb1 += dz1

        for g in [gw1, gb1, gw2, gb2, gw3, gb3]:
            g /= T

        self.w1 -= self.LR * gw1
        self.b1 -= self.LR * gb1
        self.w2 -= self.LR * gw2
        self.b2 -= self.LR * gb2
        self.w3 -= self.LR * gw3
        self.b3 -= self.LR * gb3

