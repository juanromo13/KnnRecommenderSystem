import numpy as np


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))