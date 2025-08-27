import numpy as np

#ReLU function that outpus max(0, value)
def ReLU(layer):
    return np.maximum(0,layer)

class ValueFunction:
    def __init__(self):
        self.INPUT_SIZE = 8
        self.HIDDEN_LAYER_SIZE = 20
        self.OUTPUT_SIZE = 1
        self.DISCOUNT_FACTOR = 0.9
        self.GAE_PARAM = 0.95
        self.LEARNING_RATE = 3e-4
        self.epsilon = 0.2
        self.w1 = np.random.randn(self.HIDDEN_LAYER_SIZE, self.INPUT_SIZE) * np.sqrt(2.0 / self.INPUT_SIZE)
        self.b1 = np.zeros(self.HIDDEN_LAYER_SIZE)
        self.w2 = np.random.randn(self.OUTPUT_SIZE, self.HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / self.HIDDEN_LAYER_SIZE)
        self.b2 = np.zeros(self.OUTPUT_SIZE)

    # Simple feed forward with 1 hidden layer having ReLU as activation function
    # The output is just one value, no activation function
    # This is a regression function that outputs the value to train the policy
    def forward(self, input_layer):
        z1 = self.w1 @ input_layer + self.b1
        a1 = ReLU(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = float(z2[0])
        return a2

    def set_v_old(self, states):
        self.v_old = [self.forward(s) for s in states]

    # We are given env_params that is a list of n states and their rewards of tuples: [(state,reward,action)]
    def calculate_delta_and_loss(self, env_params):
        if self.v_old is None:
            raise ValueError("v_old not set. Call set_v_old(states_0_to_T) before training.")
        n = len(env_params)
        if len(self.v_old) != n + 1:
            raise ValueError("v_old length must be len(env_params)+1 (need s_T for bootstrap).")
        delta_list = [0] * n

        for i in range(n):
            y_t = env_params[i][1] + self.DISCOUNT_FACTOR * self.v_old[i+1]
            delta_list[i] = y_t - self.v_old[i]
        return delta_list

    # We are given env_params that is a list of n states and their rewards of tuples: [(state,reward,action)] and probability ratios list r_t
    def update(self,env_params,r_t):
        delta_list = self.calculate_delta_and_loss(env_params)
        gamma, lam = self.DISCOUNT_FACTOR, self.GAE_PARAM
        T = len(delta_list)
        A = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in range(T-1, -1, -1):
            gae = delta_list[t] + gamma * lam * gae
            A[t] = gae
        A = (A - A.mean()) / (A.std() + 1e-8)

        r = np.asarray(r_t, dtype=np.float64)
        eps = self.epsilon
        unclipped_pos = (A >= 0) & (r <= 1.0 + eps)
        unclipped_neg = (A < 0)  & (r >= 1.0 - eps)
        mask = (unclipped_pos | unclipped_neg).astype(np.float64)
        g = -(A * r * mask) / T

        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = np.zeros_like(self.b2)
            
        n = len(env_params)
        for t in range(n):
            state_t, reward_t,_ = env_params[t]
            z1 = self.w1 @ state_t + self.b1
            a1 = ReLU(z1)
            v_t = float(self.w2 @ a1 + self.b2)
            y_t = reward_t + gamma * self.v_old[t+1]
            dL_dv = 2.0 * (v_t - y_t)
            grad_w2 += dL_dv * a1[None, :]
            grad_b2 += dL_dv
            relu_mask = (z1 > 0).astype(z1.dtype)
            dz1 = dL_dv * (self.w2[0] * relu_mask)
            grad_w1 += np.outer(dz1, state_t)
            grad_b1 += dz1
        grad_w1 /= n; grad_b1 /= n; grad_w2 /= n; grad_b2 /= n
        self.w1 -= self.LEARNING_RATE * grad_w1
        self.b1 -= self.LEARNING_RATE * grad_b1
        self.w2 -= self.LEARNING_RATE * grad_w2
        self.b2 -= self.LEARNING_RATE * grad_b2
        policy_loss_sum = 0
        for t in range(len(A)):
            policy_loss_sum += np.minimum(r_t[t]*A[t], np.clip(r_t[t], 1 - self.epsilon, 1 + self.epsilon) * A[t])
        return -policy_loss_sum / len(A),g
