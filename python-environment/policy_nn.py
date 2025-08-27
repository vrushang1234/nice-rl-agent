import numpy as np

#ReLU function that outpus max(0, value)
def ReLU(layer):
    return np.maximum(0,layer)

#Softmax function used to quantize the values between 0 and 1
def Softmax(layer):
    e_x = np.exp(layer - np.max(layer))
    return e_x/e_x.sum()

class PolicyNeuralNet:
    def __init__(self):
        INPUT_SIZE = 8
        HIDDEN_LAYER_SIZE = 20
        OUTPUT_SIZE = 11
        self.LR = 0.01
        self.w1 = np.random.randn(HIDDEN_LAYER_SIZE, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_LAYER_SIZE)
        self.w2 = np.random.randn(OUTPUT_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

    # Simple feed forward with 1 hidden layer having ReLU as activation function
    # The output is quantized with softmax function and it output the probabilities for nice values that can be used (might change)
    def forward(self, input_layer):
        z1 = self.w1 @ input_layer + self.b1
        a1 = ReLU(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = Softmax(z2)
        return a2

    def policy_backprop_step(self,env_params, g):
        lr = self.LR
        states = np.asarray([state[0] for state in env_params])
        actions = np.asarray([state[2] for state in env_params])
        T = states.shape[0]
        Z1 = states @ self.w1.T + self.b1          
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.w2.T + self.b2              
        m = np.max(Z2, axis=1, keepdims=True)
        P = np.exp(Z2 - m); P /= (P.sum(axis=1, keepdims=True) + 1e-8)

        Gz2 = (-g)[:, None] * P
        Gz2[np.arange(T), actions] += g

        grad_w2 = Gz2.T @ A1                           
        grad_b2 = Gz2.sum(axis=0)                      
        Ga1 = Gz2 @ self.w2                          
        Gz1 = Ga1 * (Z1 > 0)                           
        grad_w1 = Gz1.T @ states                       
        grad_b1 = Gz1.sum(axis=0)                      

        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1

