from .Connection import Connection
import NN.Activation


class Neuron:
    def __init__(self, activation):
        self.dendrites = []
        self.axons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
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
        output = 0.0
        for d in self.dendrites:
            output += d.neuron.output * d.weight
        self.output = self.activation(output)

    def get_error(self, loss, y):
        self.error = loss(self.output, y)
        # print('We got ', self.output, ' When we need ', y)

    def back_propagate(self):
        self.gradient = self.error * self.d_activation(self.output)
        lr = 0.01
        eta = 0.001
        for d in self.dendrites:
            # d.weight = d.weight + lr * self.gradient
            # d.neuron.error = d.weight * self.gradient
            d.dWeight = eta * (
            d.neuron.output * self.gradient) + lr * d.dWeight
            d.weight = d.weight - d.dWeight
            d.neuron.error += (d.weight * self.gradient)
        self.error = 0
