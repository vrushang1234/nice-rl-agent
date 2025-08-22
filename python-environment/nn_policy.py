import numpy as np

#ReLU Activation Function
def ReLU(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

#Softmax Activation Function
def softmax(x):
    x = np.array(x, dtype=np.float32)
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    den = np.sum(exp_x)
    ans = np.zeros_like(exp_x)
    for i in range(len(x)):
        ans[i] = exp_x[i] / den
    return ans

# Neural Network forward pass
class NeuralNet:
    def __init__(self, input_size=2, hidden_size=15, output_size=11, lr=0.01):
        self.lr = lr
        self.W1 = np.random.uniform(-0.01, 0.01, (hidden_size, input_size)).astype(np.float32)
        self.b1 = np.zeros((hidden_size, 1), dtype=np.float32)
        self.W2 = np.random.uniform(-0.01, 0.01, (output_size, hidden_size)).astype(np.float32)
        self.b2 = np.zeros((output_size, 1), dtype=np.float32)
        self.baseline = 0.0  
    # Forward Propogation   
    def forward(self, state):
        self.z1 = np.dot(self.W1, state) + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    # Select a Random action
    # I am doing this for now so that the RL explores all the choices, cruical for training it
    def select_action(self, state):
        probs = self.forward(state)
        action_idx = np.random.choice(len(probs), p=probs.ravel())
        nice_value = action_idx - 5
        return action_idx, nice_value

    # Back propogation to update the Weights and Biases based on the reward received from the RL agent
    # I am using a baseline value to calculate the advantage which is then used in back-prop 
    def update(self, state, action_idx, reward):
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        advantage = reward - self.baseline

        grad_z2 = self.probs.copy()
        grad_z2[action_idx] -= 1.0
        grad_z2 *= advantage

        grad_W2 = np.dot(grad_z2, self.a1.T)
        grad_b2 = grad_z2

        grad_a1 = np.dot(self.W2.T, grad_z2)
        grad_z1 = grad_a1 * relu_deriv(self.z1)

        grad_W1 = np.dot(grad_z1, state.T)
        grad_b1 = grad_z1

        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1
        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2

