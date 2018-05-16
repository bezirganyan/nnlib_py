from .Neuron import Neuron
from copy import deepcopy

class Layer:
    def __init__(self, neurons_num, activation, optimizer, input_dim=None):
        self.neurons_num = neurons_num
        self.activation = activation
        self.optimizer = optimizer
        self.neurons = []
        self.input_dim = input_dim
        self.create_neurons()

    def create_neurons(self):
        for i in range(self.neurons_num):
            self.neurons.append(Neuron(self.activation, deepcopy(self.optimizer)))

    def add_connection(self, layer):
        dendrites = layer.get_neurons()
        for n in self.neurons:
            for d in dendrites:
                n.add_dendrite(d)
                d.add_axon(n)

    def get_neurons(self):
        return self.neurons
