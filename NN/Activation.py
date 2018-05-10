import numpy as np


def sigmoid(x):
    if x >= np.log(np.finfo(type(x)).max):
        x = np.log(np.finfo(type(x)).max)
    return 1 / (np.exp(-x) + 1)


def sigmoid_der(x):
    return x * (1 - x)
    # The real formula is sigmoid(x) * (1 - sigmoid(x)),
    # but we the x given will already contain the sigmoid


def relu(x):
    return max(0, x)


def relu_der(x):
    return 0 if x <= 0 else 1


def identity(x):
    return x


def identity_der(x):
    return 1


def alias(al):
    aliases = {'sigmoid': sigmoid, 'relu': relu, 'identit': identity, '': identity,
               'sigmoid_der': sigmoid_der, 'relu_der': relu_der,
               'identity_det': identity_der, '_der': identity_der}
    return aliases[al]
