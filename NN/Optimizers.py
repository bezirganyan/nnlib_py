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
            d.dWeight = self.momentum * d.dWeight -\
                        self.lr * (d.neuron.output * self.gradient)

            if self.nesterov:
                # d.weight = d.weight - d.dWeight
                d.weight = d.weight + self.momentum * d.dWeight -\
                           self.lr * (d.neuron.output * self.gradient)
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


class RMSProp(Optimizer):
    def __init__(self, rho=0.9, lr=0.001, epsilon=1e-7):
        super().__init__()
        self.rho = rho
        self.lr = lr
        self.epsilon = epsilon
        self.hist_grad = 0

    def optimize(self, dendrites, neuron):
        self.dendrites = dendrites
        self.neuron = neuron

        self.gradient = self.neuron.error * self.neuron.d_activation(self.neuron.output)

        self.hist_grad = self.hist_grad * self.rho + (1. - self.rho) * self.gradient ** 2
        rate_change = self.gradient / (self.epsilon + np.sqrt(self.hist_grad))
        update = self.lr * rate_change

        for d in self.dendrites:
            d.weight = d.weight - update
            d.neuron.error += np.mean((d.weight * self.gradient))
    # TODO - Check if possible


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.ms = None
        self.vs = None
        self.t = 0

    def optimize(self, dendrites, neuron):
        self.dendrites = dendrites
        self.neuron = neuron

        self.t += 1
        if self.ms is None:
            self.ms = np.zeros(np.shape(self.dendrites[0].neuron.output))
            self.vs = np.zeros(np.shape(self.dendrites[0].neuron.output))

        self.gradient = self.neuron.error * self.neuron.d_activation(self.neuron.output)
        lr_t = self.lr * (np.sqrt(1. - np.power(self.beta_2, self.t)) /
                     (1. - np.power(self.beta_1, self.t)))

        self.ms = self.beta_1 * self.ms + (1 - self.beta_1) * self.gradient
        self.vs = self.beta_2 * self.vs + (1 - self.beta_2) * np.square(self.gradient)

        # m_hat = self.ms / (1 - np.power(self.beta_1, self.t))
        # v_hat = self.vs / (1 - np.power(self.beta_2, self.t))

        update = lr_t * self.ms / (np.sqrt(self.vs) + self.epsilon)

        for d in self.dendrites:
            d.weight = d.weight - update
            d.neuron.error += np.mean((d.weight * self.gradient))

        # TODO - Check if possible
