import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7    # to prevent log(0) -inf

    return -np.sum(t * np.log(y + delta)) / batch_size  # one hot label (e.g., 0 0 0 1 0) -> 3

    # return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size    # when it is not one hot label (e.g., 3)