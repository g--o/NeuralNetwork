#### example of training methods

from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot

import DataSet
import PerformanceFunction

DEFAULT_BACKPROP_LEARN_RATE = -0.1
DEFAULT_BACKPROP_MAX_ITERS  = 1000

class TrainerDebug:
    enabled = True
    errors_log = []
    debug_deltas = None

    @staticmethod
    def draw_error():
        pyplot.plot(TrainerDebug.errors_log)
        pyplot.show()

    @staticmethod
    def update(iter, neural_network, w_deltas, data_output, error_fn):
        if TrainerDebug.enabled:
            # log errors
            TrainerDebug.errors_log += [error_fn(neural_network, data_output)]
            if iter == 0:
                TrainerDebug.debug_deltas = w_deltas

class Trainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, neural_network, data_set):
        pass

class Backpropagation(Trainer):
    def __init__(self,  max_iterations = DEFAULT_BACKPROP_MAX_ITERS,
                        learn_rate = DEFAULT_BACKPROP_LEARN_RATE,
                        performance_fn = PerformanceFunction.SmoothDelta):
        self.max_iterations = max_iterations
        self.learn_rate = learn_rate
        self.performance = performance_fn.performance
        self.dperformance = performance_fn.dperformance

    def train(self, neural_network, data_set):
        """Train a neural network with data_set."""
        # Initialize deltas
        l_deltas = neural_network.l[:]
        w_deltas = neural_network.w[:]

        for i in xrange(self.max_iterations):
            sample = data_set.getSample()
            data_input = sample.inputs
            data_output = sample.outputs
            # feed the data set
            neural_network.feed(data_input)
            # set deltas - get last layer error
            last_layer_index = neural_network.num_layers-1
            l_deltas[last_layer_index] = self.get_layer_deltas(neural_network, last_layer_index, prediction = data_output)
            w_deltas[last_layer_index] = np.zeros_like(neural_network.w[last_layer_index])
            # propagate error
            for j in xrange(last_layer_index-1, -1, -1):
                w_deltas[j] = self.get_weights_deltas(neural_network.l[j], l_deltas[j+1])
                if (j != 0): # first layer = input layer
                    l_deltas[j] = self.get_layer_deltas(neural_network, j, next_layer_deltas=l_deltas[j+1], prediction = data_output)
            # adjust weights
            neural_network.adjust_weights(w_deltas)
            TrainerDebug.update(i, neural_network, w_deltas, data_output, self.get_average_error)

    def get_weights_deltas(self, layer, next_layer_deltas):
        return self.learn_rate * np.dot(layer.T, next_layer_deltas)

    def get_layer_deltas(self, neural_network, layer_index, next_layer_deltas = None, prediction = None):
        preactive_layer = np.dot(neural_network.l[layer_index - 1], neural_network.w[layer_index - 1])

        if layer_index == neural_network.num_layers-1: # last layer = output layer
            layer = neural_network.l[layer_index]
            return self.dperformance(prediction, layer) * neural_network.dactivation(preactive_layer)
        else: # hidden layer
            layer_weights = neural_network.w[layer_index]
            return self.get_deltas(neural_network, preactive_layer, layer_weights, next_layer_deltas)

    def get_deltas(self, neural_network, preactive_layer, layer_weights, next_layer_deltas):
        if next_layer_deltas is None:
            print "ERROR: failed to get deltas - bad next_layer_deltas."
            return None
        transposed_weights = np.array(layer_weights).T
        return np.dot(next_layer_deltas, transposed_weights) * neural_network.dactivation(preactive_layer)

    def get_average_error(self, neural_network, prediction):
        last_layer = neural_network.l[neural_network.num_layers-1]
        last_layer_err = self.performance(prediction, last_layer)
        return (np.mean(np.abs(last_layer_err)))
