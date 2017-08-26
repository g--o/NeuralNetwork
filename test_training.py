
from config import *

def test_training():
    config = experiment_config
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
    print "error: ", trainer.get_average_error(neural_network, outMatrix)

    # visual output
    if PLOT_NEURAL_NET:
        network = Visualizer.DrawNN([DEFAULT_INPUT_DIM[0]] + [config[0]]*config[1] + [DEFAULT_OUTPUT_DIM[0]])
        network.draw()
    if PLOT_ERROR_LOG:
        trainer.draw_error()
