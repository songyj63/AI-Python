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
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y