#### example of training methods

import numpy as np

import DataSet
import PerformanceFunction
from matplotlib import pyplot

DEFAULT_BACKPROP_LEARN_RATE = -0.1
DEFAULT_BACKPROP_MAX_ITERS  = 1000

class Trainer(object):
    pass

class Backpropagation(Trainer):
    def __init__(self,  max_iterations = DEFAULT_BACKPROP_MAX_ITERS,
                        learn_rate = DEFAULT_BACKPROP_LEARN_RATE,
                        performance_fn = PerformanceFunction.SmoothDelta):
        self.max_iterations = max_iterations
        self.learn_rate = learn_rate
        self.performance = performance_fn.performance
        self.dperformance = performance_fn.dperformance
        self.errors_log = []

    def train(self, neural_network, data_set):
        """Train a neural network with data_set."""
        self.errors_log = []
        for i in xrange(self.max_iterations):
            data_input = data_set.inputs
            data_output = data_set.outputs
            # Reset calculated deltas
            l_deltas = neural_network.l[:]
            w_deltas = neural_network.w[:]
            # feed the data set
            neural_network.feed(data_input)
            # get last layer error
            last_layer_index = neural_network.num_layers-1
            last_layer = neural_network.l[last_layer_index]
            l_deltas[last_layer_index] = self.get_layer_deltas(neural_network, last_layer_index, prediction = data_output)
            # propagate error
            for j in xrange(last_layer_index-1, -1, -1):
                w_deltas[j] = self.get_weights_deltas(neural_network.l[j], l_deltas[j+1])
                if (j != 0): # first layer = input layer
                    l_deltas[j] = self.get_layer_deltas(neural_network, j, next_layer_deltas=l_deltas[j+1], prediction = data_output)
            # adjust weights
            neural_network.adjust_weights(w_deltas)
            # log errors
            if i % self.max_iterations/3.0:
                self.errors_log += [self.get_average_error(neural_network, data_output)]

    def get_weights_deltas(self, layer, next_layer_deltas):
        return self.learn_rate * np.dot(layer.T, next_layer_deltas)

    def get_layer_deltas(self, neural_network, layer_index, next_layer_deltas = None, prediction = None):
        if neural_network.num_layers-1 == layer_index: # last layer = output layer
            layer = neural_network.l[layer_index]
            return self.dperformance(prediction, layer) # * neural_network.dactivation(layer)
        else: # hidden layer
            layer_weights = neural_network.w[layer_index]
            layer = neural_network.l[layer_index]
            return self.get_deltas(neural_network, layer, layer_weights, next_layer_deltas)

    def get_deltas(self, neural_network, layer, layer_weights, next_layer_deltas):
        if next_layer_deltas is None:
            print "ERROR: failed to get deltas - bad next_layer_deltas."
            return None
        transposed_weights = np.array(layer_weights).T
        return np.dot(next_layer_deltas, transposed_weights) * neural_network.dactivation(layer)

    def get_average_error(self, neural_network, prediction):
        last_layer = neural_network.l[neural_network.num_layers-1]
        last_layer_err = prediction - last_layer
        return (np.mean(np.abs(last_layer_err)))

    def draw_error(self):
        pyplot.plot(self.errors_log)
        pyplot.show()
