import sys
import copy
import numpy as np
sys.path.append("core")

import NeuralNetwork
import Trainer
import DataSet

import ActivationFunction
import Visualizer

############### tests configurations ###############
# general
PLOT_NEURAL_NET = True
PLOT_ERROR_LOG = True
# xor training
inMatrix = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=np.float64)
outMatrix = np.array(([0],[1],[1],[0]), dtype=np.float64)

############### neural network configurations ###############
# test data set
DEFAULT_DATA_SET = DataSet.DataSet(inMatrix, outMatrix)
# dimentions = (node_size, num_nodes)
DEFAULT_INPUT_DIM = inMatrix.T.shape
DEFAULT_OUTPUT_DIM = outMatrix.T.shape
# training configurations (hidden_width, num_layers, max_iterations, learn_rate)
grad_test_config = (5, 1, 1, -1)
experiment_config = (5, 1, 10000, -1)
simple_config = (5, 1, 20000, -0.8202)
double_config = (5, 2, 10000, -0.8202)

def print_title(s):
    print "=" * 5, s, "="*5

def create_training_setup(config):
    # create setup from configuration
    print_title("creating setup")
    hidden_width, num_layers, max_iterations, learn_rate = config
    print "Configured!"
    # make neural network
    neural_network = NeuralNetwork.NeuralNetwork(DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, hidden_width, num_layers)
    print "Neural network initialized!"
    # make a trainer
    trainer = Trainer.Backpropagation(max_iterations = max_iterations, learn_rate = learn_rate)
    print "Trainer initialized!"

    return neural_network, trainer
