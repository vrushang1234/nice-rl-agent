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

    # We are given env_params that is a list of n states and their rewards of tuples: [(state,reward)]
    def calculate_delta(self, env_params):
        v_t_list = [self.forward(env_params[0][0])] * len(env_params)
        delta_list = [0] * len(env_params)
        for i in range(len(env_params)-1):
            v_t_list[i+1] = self.forward(env_params[i+1][0])
            y_t = env_params[i][1] + self.DISCOUNT_FACTOR * v_t_list[i+1]
            delta_list[i] = y_t - v_t_list[i]
        delta_list[-1] = env_params[-1][1] - v_t_list[-1]
        return delta_list


    # We are given env_params that is a list of n states and their rewards of tuples: [(state,reward)]
    def update(self,env_params):
        delta_list = self.calculate_delta(env_params)
        prod = self.DISCOUNT_FACTOR * self.GAE_PARAM
        adv_0 = 0
        for i in range(len(delta_list)):
            adv_0 += delta_list[i] * pow(prod,i)
        adv_list = [adv_0]
        for i in range(1,len(delta_list)):
            adv_list.append((adv_list[i-1] - delta_list[i-1])/prod)
        

        #TODO: Update the parameters with loss and update the policy with adv

