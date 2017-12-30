import numpy as np
import random

class DataSet(object):
    def __init__(self, inputs = None, outputs = None):
        self.inputs = inputs
        self.outputs = outputs

    def getSample(self):
        sample_index = random.randint(0, len(self.outputs) - 1)
        ins = np.array((self.inputs[sample_index],), dtype=np.float64)
        outs = np.array((self.outputs[sample_index],), dtype=np.float64)
        return DataSet(ins, outs)
