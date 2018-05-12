import numpy as np


class Optimizer(object):
    def __init__(self):
        self.dendrites = []
        self.neuron = None
        self.gradient = 0.0

    def optimize(self, dendrites, neuron):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.001, momentum=0.5, nesterov=False):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, dendrites, neuron):
        self.dendrites = dendrites
        self.neuron = neuron

        self.gradient = self.neuron.error * self.neuron.d_activation(self.neuron.output)

        for d in self.dendrites:
            d.dWeight = self.momentum * d.dWeight - self.lr * (d.neuron.output * self.gradient)

            if self.nesterov:
                # d.weight = d.weight - d.dWeight
                d.weight = d.weight + self.momentum * d.dWeight - self.lr*(d.neuron.output * self.gradient)
            else:
                d.weight = d.weight + d.dWeight

            d.neuron.error += np.mean((d.weight * self.gradient))

    # TODO - Check if possible


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.hist_grad = None
        self.epsilon = epsilon

    def optimize(self, dendrites, neuron):
        self.dendrites = dendrites
        self.neuron = neuron

        self.gradient = self.neuron.error * self.neuron.d_activation(self.neuron.output)
        if self.hist_grad is None:
            self.hist_grad = self.gradient ** 2
        else:
            self.hist_grad += self.gradient ** 2
        rate_change = self.gradient / (self.epsilon + np.sqrt(self.hist_grad))
        update = self.lr * rate_change
        for d in self.dendrites:
            d.weight = d.weight - update
            d.neuron.error += np.mean((d.weight * self.gradient))
