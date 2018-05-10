import numpy as np


class Connection:
    def __init__(self, neuron):
        self.neuron = neuron
        self.weight = np.random.normal()
        self.dWeight = 0.0 # TODO - check if needed