import numpy as np
from matplotlib import pyplot as plt

import NeuralNetwork
import Trainer
import DataSet

import ActivationFunction
import Visualizer

# xor training
inMatrix = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=np.float64)
outMatrix = np.array(([0],[1],[1],[0]), dtype=np.float64)

# configurations
# in (node_size, num_nodes)
INPUT_DIM = inMatrix.T.shape
OUTPUT_DIM = outMatrix.T.shape
DATA_SET = DataSet.DataSet(inMatrix, outMatrix)

# (hidden_width, num_layers, max_iterations, learn_rate)
simple_config = (5, 1, 20000, -0.8202)
double_config = (5, 2, 10000, -0.8202)

def main():
    hidden_width, num_layers, max_iterations, learn_rate = simple_config
    print "Intializing!"
    # make neural network
    neural_network = NeuralNetwork.NeuralNetwork(INPUT_DIM, OUTPUT_DIM, hidden_width, num_layers)
    print "Neural network initialized!"
    # make a trainer
    trainer = Trainer.Backpropagation(max_iterations = max_iterations, learn_rate = learn_rate)
    print "Trainer initialized!"
    # start training
    trainer.train(neural_network, DATA_SET)
    print "Done training!"
    # test
    neural_network.feed(inMatrix)
    print "output: "
    print neural_network.get_output()
    print "expected: "
    print outMatrix
    print "error: ", trainer.get_average_error(neural_network, outMatrix)
    # visualizer
    trainer.draw_error()
    network = Visualizer.DrawNN([INPUT_DIM[0]] + [hidden_width]*num_layers + [OUTPUT_DIM[0]])
    network.draw()

# entry point
main()
