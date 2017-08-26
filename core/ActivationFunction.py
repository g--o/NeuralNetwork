#### example of activation functions

import numpy as np

class ActivationFunction(object):
    pass

class ReLU(ActivationFunction):
    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def dactivation(x):
        return (x > 0)

class Sigmoid(ActivationFunction):
    @staticmethod
    def activation(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def dactivation(x):
        y = Sigmoid.activation(x)
        return y * (1-y)
