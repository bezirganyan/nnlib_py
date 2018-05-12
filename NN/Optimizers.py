import numpy as np


class sdg:
    def __init__(self, lr=0.01, momentum=0.001):
        # self.batch_size = batch_size
        self.lr = lr
        self.eta = momentum
        self.dendrites = []
        self.neuron = None
        self.gradient = 0.0

    def optimize(self, dendrites, neuron):
        self.dendrites = dendrites
        self.neuron = neuron

        self.gradient = self.neuron.error * self.neuron.d_activation(self.neuron.output)

        for d in self.dendrites:
            d.dWeight = self.eta * (d.neuron.output * self.gradient) + self.lr * d.dWeight
            d.weight = d.weight - d.dWeight
            d.neuron.error += np.mean((d.weight * self.gradient))