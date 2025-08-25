import numpy as np

#ReLU function that outpus max(0, value)
def ReLU(layer):
    return np.maximum(0,layer)

#Softmax function used to quantize the values between 0 and 1
def Softmax(layer):
    e_x = np.exp(layer - np.max(layer))
    return e_x/e_x.sum()

class NeuralNet:
    def __init__(self):
        INPUT_SIZE = 8
        HIDDEN_LAYER_SIZE = 20
        OUTPUT_SIZE = 11
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

