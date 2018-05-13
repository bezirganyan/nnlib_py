import numpy as np


def lse(x, y):
    err = (y - x)**2
    return np.mean(err)
