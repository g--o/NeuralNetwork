


from test_training import test_training
from test_gradiant import test_gradiant

#### output configurations
NUM_LEARN_TESTS = 1
NUM_GRAD_TESTS = 1

def main():
    for i in xrange(NUM_GRAD_TESTS):
        test_gradiant()
    for i in xrange(NUM_LEARN_TESTS):
        test_training()
# entry point
main()
