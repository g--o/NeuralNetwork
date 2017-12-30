import DataSet
from config import *

DS_INPUT = np.array(([0,0], [1,0], [0,1], [1,1]), dtype=np.float64)
DS_OUTPUT = np.array(([0], [1], [1], [0]), dtype=np.float64)
DATA_SET = DataSet.DataSet(DS_INPUT, DS_OUTPUT)

def test_primes():
    print DATA_SET
    config = simple_config
    # get setup
    neural_network, trainer = create_training_setup(config, (1, 2), (1, 1))
    # start training
    print_title("test primes")
    trainer.train(neural_network, DATA_SET)
    # test
    test_sample = DATA_SET.getSample()
    neural_network.feed(test_sample.inputs)
    print "output: "
    print neural_network.get_output()
    print "expected: "
    print outMatrix
    print "error: ", trainer.get_average_error(neural_network, test_sample.outputs)

    # visual output
    if PLOT_NEURAL_NET:
        network = Visualizer.DrawNN([DEFAULT_INPUT_DIM[0]] + [config[0]]*config[1] + [DEFAULT_OUTPUT_DIM[0]])
        network.draw()
    if PLOT_ERROR_LOG:
        trainer.draw_error()
