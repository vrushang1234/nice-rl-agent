import numpy as np

#ReLU function that outpus max(0, value)
def ReLU(layer):
    return np.maximum(0,layer)

class NeuralNet:
    def __init__(self):
        INPUT_SIZE = 8
        HIDDEN_LAYER_SIZE = 20
        OUTPUT_SIZE = 1
        self.w1 = np.random.randn(HIDDEN_LAYER_SIZE, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_LAYER_SIZE)
        self.w2 = np.random.randn(OUTPUT_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

    # Simple feed forward with 1 hidden layer having ReLU as activation function
    # The output is just one value, no activation function
    # This is a regression function that outputs the value to train the policy
    def forward(self, input_layer):
        z1 = self.w1 @ input_layer + self.b1
        a1 = ReLU(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = float(z2[0])
        return a2

