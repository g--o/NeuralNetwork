
from config import *

MAX_RATIO = 1e-08
EPSILON = 1e-04
grad_test_config = (5, 1, 1, -1, DEFAULT_ACTIVATION_FN)

############### test configuration ###############
inMatrix = np.array(([0,0],), dtype=np.float64)
outMatrix = np.array(([0],), dtype=np.float64)
# test data set
DATA_SET = DataSet.DataSet(inMatrix, outMatrix)
# dimentions = (node_size, num_nodes)
INPUT_DIM = inMatrix.T.shape
OUTPUT_DIM = outMatrix.T.shape

def flatten(arr):
    result = []
    for arr in [y.flatten() for y in arr]:
        result += arr.tolist()
    return result

def test_gradiant():
    neural_network, trainer = create_training_setup(grad_test_config, INPUT_DIM, OUTPUT_DIM)

    e = EPSILON
    numgrad = copy.deepcopy(neural_network.w)

    for i in xrange(len(numgrad)): # for each layer except last
        for j in xrange(len(numgrad[i])): # for each node
            for k in xrange(len(numgrad[i][j])): # for each weight
                # compute w+e
                neural_network.w[i][j][k] += e
                neural_network.feed(inMatrix)
                loss1 = trainer.performance(outMatrix, neural_network.get_output())
                # compute w-e
                neural_network.w[i][j][k] -= 2*e
                neural_network.feed(inMatrix)
                loss2 = trainer.performance(outMatrix, neural_network.get_output())
                # get slope/derivative
                numgrad[i][j][k] = sum(loss2 - loss1) / (2*e)
                neural_network.w[i][j][k] += e
            numgrad[i][j] = np.array(numgrad[i][j]).ravel()

    # check Backpropagation gradiant
    trainer.train(neural_network, DATA_SET)

    # calculate score - lower the better
    debug_deltas = np.array(get_debug_deltas())
    sub = flatten(debug_deltas - np.array(numgrad))
    add = flatten(np.array(numgrad) + debug_deltas)

    ratio = np.linalg.norm(sub)/np.linalg.norm(add)
    print "maximum ratio: ", MAX_RATIO
    print "ratio: ", ratio
    print_result(ratio < MAX_RATIO)
