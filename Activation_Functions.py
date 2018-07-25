import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid_function(x):
    return 1/(1+np.exp(-x))


def relu_function(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax_function(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))