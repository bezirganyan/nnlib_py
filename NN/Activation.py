import numpy as np


def sigmoid(x):
    # if x >= np.log(np.finfo(type(x.any())).max):
    #     x = np.log(np.finfo(type(x)).max)
    return 1 / (np.exp(-x) + 1)


def sigmoid_der(x):
    return x * (1 - x)
    # The real formula is sigmoid(x) * (1 - sigmoid(x)),
    # but we the x given will already contain the sigmoid


def relu(x):
    return np.maximum(x, 0, x)


def relu_der(x):
    return np.array([0 if n <= 0 else 1 for n in x])


def identity(x):
    return x


def identity_der(x):
    return np.ones(5)


def alias(al):
    aliases = {'sigmoid': sigmoid, 'relu': relu, 'identit': identity, '': identity,
               'sigmoid_der': sigmoid_der, 'relu_der': relu_der,
               'identity_det': identity_der, '_der': identity_der}
    return aliases[al]
