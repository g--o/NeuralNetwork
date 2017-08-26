
from config import *

def flatten(arr):
    result = []
    for arr in [y.flatten() for y in arr]:
        result += arr.tolist()
    return result

def test_gradiant():
    neural_network, trainer = create_training_setup(grad_test_config)

    e = 1e-04
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
    trainer.train(neural_network, DEFAULT_DATA_SET)

    # calculate score - lower the better
    sub = flatten(np.array(trainer.debug_deltas) - np.array(numgrad))
    add = flatten(np.array(numgrad) + np.array(trainer.debug_deltas))

    print np.linalg.norm(sub)/np.linalg.norm(add)
