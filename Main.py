import numpy as np
import matplotlib.pylab as plt
import Perceptron
import Activation_Functions as af
import NeuralNet as nn


# print(Perceptron.XOR(0,0))
# print(Perceptron.XOR(1,0))
# print(Perceptron.XOR(0,1))
# print(Perceptron.XOR(1,1))

# # draw step function
# x = np.arange(-5.0, 5.0, 0.1)
# y = af.step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# # draw sigmoid function
# x = np.arange(-5.0, 5.0, 0.1)
# y = af.sigmoid_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# # neural network basic
# network = nn.init_network()
# x = np.array([1.0, 0.5])
# y = nn.forward(network, x)
# print(y)

a = np.array([0.3, 2.9, 4.0])
print(af.softmax_function(a))

print("test")