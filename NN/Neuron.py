from .Connection import Connection
import NN.Activation
import numpy as np


class Neuron:
    def __init__(self, activation):
        self.dendrites = []
        self.axons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = None
        if type(activation) is str:
            self.activation = NN.Activation.alias(activation)
            self.d_activation = NN.Activation.alias(activation + '_der')
        else:
            raise ValueError('Wrong value for activation function: Expected str')

    def add_dendrite(self, neuron):
        self.dendrites.append(Connection(neuron))

    def add_axon(self, neuron):
        self.axons.append(Connection(Neuron))

    def feed_forward(self):
        output = np.zeros(np.shape(self.dendrites[0].neuron.output))
        for d in self.dendrites:
            output += d.neuron.output * d.weight
        self.output = self.activation(output)

    def get_error(self, loss, y):
        self.error = loss(self.output, y)
        # print('We got ', self.output, ' When we need ', y)

    def back_propagate(self, optimizer):
        optimizer.optimize(self.dendrites, self)
        self.error = 0
