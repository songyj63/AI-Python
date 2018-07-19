import numpy as np
from Activation_Functions import softmax_function
from Loss_Functions import cross_entropy_error
from Gradient_Functions import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_function(z)
        loss = cross_entropy_error(y, t)

        return loss

