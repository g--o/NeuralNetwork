# standard libs
import sys
import copy
import numpy as np

# Neural network core
sys.path.append("core")
import NeuralNetwork
import Trainer
import DataSet
import ActivationFunction

# Presentation
import Visualizer
from utils import *

############### tests configurations ###############
# general
PLOT_NEURAL_NET = True
PLOT_ERROR_LOG = True
# xor training
inMatrix = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=np.float64)
outMatrix = np.array(([0],[1],[1],[0]), dtype=np.float64)

############### neural network configurations ###############
DEFAULT_ACTIVATION_FN = ActivationFunction.Sigmoid
# test data set
DEFAULT_DATA_SET = DataSet.DataSet(inMatrix, outMatrix)
# dimentions = (node_size, num_nodes)
DEFAULT_INPUT_DIM = inMatrix.T.shape
DEFAULT_OUTPUT_DIM = outMatrix.T.shape
# training configurations (hidden_width, num_layers, max_iterations, learn_rate)
experiment_config = (5, 1, 10000, -1, DEFAULT_ACTIVATION_FN)
simple_config = (5, 1, 20000, -0.8202, DEFAULT_ACTIVATION_FN)
double_config = (5, 2, 10000, -0.8202, DEFAULT_ACTIVATION_FN)

############### testing helpers ###############

def create_training_setup(config, input_dim = DEFAULT_INPUT_DIM, output_dim = DEFAULT_OUTPUT_DIM):
    # create setup from configuration
    print_subtitle("creating setup")
    hidden_width, num_layers, max_iterations, learn_rate, activation_fn = config
    print "Configured!"
    # make neural network
    neural_network = NeuralNetwork.NeuralNetwork(input_dim, output_dim, hidden_width, num_layers, activation_fn)
    print "Neural network initialized!"
    # make a trainer
    trainer = Trainer.Backpropagation(max_iterations = max_iterations, learn_rate = learn_rate)
    print "Trainer initialized!"

    return neural_network, trainer
