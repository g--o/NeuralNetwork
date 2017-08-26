
import numpy as np

import ActivationFunction
import PerformanceFunction

###### default functions

def create_weights(input_dim, output_dim, hidden_width, num_layers):
    weights = np.random.randn(1,            input_dim[0],  hidden_width).tolist()
    if num_layers > 1:
        weights += np.random.randn(num_layers-1, hidden_width,  hidden_width).tolist()
    weights +=  np.random.randn(1,            hidden_width,  output_dim[0]).tolist() + \
                np.random.randn(1,            output_dim[0], 1).tolist()
    return weights

def create_layer(dimentions):
    return np.zeros(dimentions)

def create_layers(input_dim, output_dim, hidden_width, num_layers):
    layers = [create_layer(input_dim)]
    layers += [create_layer((hidden_width, input_dim[1]))] * num_layers
    layers += [create_layer(output_dim)]
    return layers

###### neural network

class NeuralNetwork(object):
    # Neural network class

    def __init__(self, input_dim, output_dim, hidden_width, num_layers, activation_fn = ActivationFunction.Sigmoid):
        #np.random.seed(1)
        self.l = create_layers(input_dim, output_dim, hidden_width, num_layers)
        self.w = create_weights(input_dim, output_dim, hidden_width, num_layers)

        self.activation_fn = activation_fn
        self.activation = activation_fn.activation #lambda x: [self.pre_activate(y) for y in x]
        self.dactivation = activation_fn.dactivation #lambda x: [self.pre_dactivate(y) for y in x]
        self.num_layers = num_layers + 2 # account for input & ouput layers

    def pre_dactivate(self, x):
        if hasattr(x, "__len__"):
            return [self.activation_fn.dactivation(y) for y in x]
        else:
            return self.activation_fn.dactivation(x)

    def pre_activate(self, x):
        if hasattr(x, "__len__"):
            return [self.activation_fn.activation(y) for y in x]
        else:
            return self.activation_fn.activation(x)

    def forward(self, layer, weights):
        return self.activation(np.dot(layer, weights))

    def feed(self, inputs):
        self.l[0] = inputs

        for j in xrange(1, self.num_layers):
            self.l[j] = self.forward(self.l[j-1], self.w[j-1])

    def adjust_weights(self, weights_deltas):
        """ adjust weights by adding weights_deltas """
        for layer_index in xrange(len(self.w)):
            self.w[layer_index] = np.add(self.w[layer_index], weights_deltas[layer_index])

    def get_input(self):
        return self.l[0]

    def get_output(self):
        return self.l[self.num_layers-1]
