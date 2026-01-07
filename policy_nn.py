import numpy as np
import os, csv

#ReLU function that outpus max(0, value)
def ReLU(layer):
    return np.maximum(0,layer)

#Softmax function used to quantize the values between 0 and 1
def Softmax(layer):
    e_x = np.exp(layer - np.max(layer))
    return e_x/e_x.sum()

class PolicyNeuralNet:
    def __init__(self):
        INPUT_SIZE = 6
        HIDDEN_LAYER_1_SIZE = 50 
        HIDDEN_LAYER_2_SIZE = 70 
        OUTPUT_SIZE = 11
        self.LR = 3e-4
        self.w1 = np.random.randn(HIDDEN_LAYER_1_SIZE, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_LAYER_1_SIZE)
        self.w2 = np.random.randn(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_1_SIZE)
        self.b2 = np.zeros(HIDDEN_LAYER_2_SIZE)
        self.w3 = np.random.randn(OUTPUT_SIZE, HIDDEN_LAYER_2_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_2_SIZE)
        self.b3 = np.zeros(OUTPUT_SIZE)

    # Simple feed forward with 1 hidden layer having ReLU as activation function
    # The output is quantized with softmax function and it output the probabilities for nice values that can be used (might change)
    def forward(self, input_layer):
        z1 = self.w1 @ input_layer + self.b1
        a1 = np.tanh(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = np.tanh(z2)
        z3 = self.w3 @ a2 + self.b3
        a3 = Softmax(z3)
        return a3

    def _save_params_csv(self, path="policy_params.csv"):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["param", "row", "col", "value"])

            for r in range(self.w1.shape[0]):
                for c in range(self.w1.shape[1]):
                    writer.writerow(["w1", r, c, float(self.w1[r, c])])
            for r in range(self.w2.shape[0]):
                for c in range(self.w2.shape[1]):
                    writer.writerow(["w2", r, c, float(self.w2[r, c])])
            for r in range(self.w3.shape[0]):
                for c in range(self.w3.shape[1]):
                    writer.writerow(["w3", r, c, float(self.w3[r, c])])

            for i, v in enumerate(self.b1):
                writer.writerow(["b1", i, -1, float(v)])
            for i, v in enumerate(self.b2):
                writer.writerow(["b2", i, -1, float(v)])
            for i, v in enumerate(self.b3):
                writer.writerow(["b3", i, -1, float(v)])

        os.replace(tmp, path)

    def policy_backprop_step(self, env_params, advantages, ratios, eps=0.2):
        lr = self.LR

        states = np.asarray([t[0] for t in env_params])
        actions = np.asarray([t[2] for t in env_params])
        A = np.asarray(advantages, dtype=np.float64)
        r = np.asarray(ratios, dtype=np.float64)

        T = states.shape[0]

        # PPO clipped objective coefficient
        clipped_r = np.clip(r, 1.0 - eps, 1.0 + eps)
        coeff = np.where(A >= 0, np.minimum(r, clipped_r), np.maximum(r, clipped_r))
        g = coeff * A           # g_t = dL/d(log π)

        # ----- Forward pass (batched) -----
        Z1 = states @ self.w1.T + self.b1
        A1 = np.tanh(Z1)

        Z2 = A1 @ self.w2.T + self.b2
        A2 = np.tanh(Z2)

        Z3 = A2 @ self.w3.T + self.b3
        Z3 -= np.max(Z3, axis=1, keepdims=True)

        P = np.exp(Z3)
        P /= (P.sum(axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(P * np.log(P + 1e-8), axis=1)
        g += 0.01 * entropy

        # ----- Policy gradient -----
        Gz3 = (-g)[:, None] * P
        Gz3[np.arange(T), actions] += g

        grad_w3 = Gz3.T @ A2
        grad_b3 = Gz3.sum(axis=0)

        Ga2 = Gz3 @ self.w3
        Gz2 = Ga2 * (1.0 - A2 ** 2)

        grad_w2 = Gz2.T @ A1
        grad_b2 = Gz2.sum(axis=0)

        Ga1 = Gz2 @ self.w2
        Gz1 = Ga1 * (1.0 - A1 ** 2)

        grad_w1 = Gz1.T @ states
        grad_b1 = Gz1.sum(axis=0)

        # Normalize
        grad_w1 /= T; grad_b1 /= T
        grad_w2 /= T; grad_b2 /= T
        grad_w3 /= T; grad_b3 /= T

        # Gradient clipping
        def clip_(x, max_norm=5.0):
            n = np.linalg.norm(x)
            if np.isfinite(n) and n > max_norm:
                x *= max_norm / (n + 1e-8)

        for g_ in [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]:
            clip_(g_)

        # SGD update
        self.w3 -= lr * grad_w3
        self.b3 -= lr * grad_b3
        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1

        self._save_params_csv()


