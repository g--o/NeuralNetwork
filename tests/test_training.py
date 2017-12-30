
from config import *

MAX_ERR_FOR_CONFIG = 1e-03 # temporary until bias is fixed
learning_config = simple_config

def test_training():
    config = learning_config
    # get setup
    neural_network, trainer = create_training_setup(config)
    # start training
    print_title("test training")
    trainer.train(neural_network, DEFAULT_DATA_SET)
    # test
    neural_network.feed(inMatrix)
    print "output: "
    print neural_network.get_output()
    print "expected: "
    print outMatrix
    err = trainer.get_average_error(neural_network, outMatrix)
    print "error: ", err
    print_result(err < MAX_ERR_FOR_CONFIG)

    # visual output
    if PLOT_NEURAL_NET:
        network = Visualizer.DrawNN([DEFAULT_INPUT_DIM[0]] + [config[0]]*config[1] + [DEFAULT_OUTPUT_DIM[0]])
        network.draw()
    if PLOT_ERROR_LOG:
        trainer.draw_error()
