
import sys
sys.path.append('tests')

from config import print_testname
from test_training import test_training
from test_gradiant import test_gradiant
from test_primes import test_primes

#### output configurations
NUM_LEARN_TESTS = 1
NUM_GRAD_TESTS = 1

def main():
    # test_primes()
    print_testname("Gradient Testing")
    for i in xrange(NUM_GRAD_TESTS):
        test_gradiant()

    print_testname("Learning Testing")
    for i in xrange(NUM_LEARN_TESTS):
        test_training()
# entry point
main()
