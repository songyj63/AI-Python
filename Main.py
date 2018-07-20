import numpy as np
import sys, os
from Gradient_Simplenet import simpleNet
from dataset.mnist import load_mnist
from PIL import Image
import NeuralNet_Mnist_Eaxmple as nme
import matplotlib.pylab as plt
import Gradient_Functions as fc


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

# # softmax test
# a = np.array([0.3, 2.9, 4.0])
# print(af.softmax_function(a))

# # MNIST data load and draw
# sys.path.append(os.pardir)
#
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)
#
# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
# img_show(img)

# # MNIST example, predict & accuracy
# x, t = nme.get_data()
# network = nme.init_network()
#
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = nme.predict(network, x[i])
#     p = np.argmax(y)    # 확률이 가장 높은 원소의 인덱스를 얻는다
#     if p == t[i]:
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# # MNIST example, predict & accuracy using batch
# x, t = nme.get_data()
# network = nme.init_network()
#
# batch_size = 100
# accuracy_cnt = 0
#
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i+batch_size]
#     y_batch = nme.predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# # diff example 1
# def function_1(x):
#     return 0.01*x**2 + 0.1*x
#
#
# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()
#
# print(fc.numerical_diff(function_1, 5))
# print(fc.numerical_diff(function_1, 10))

# # diff example 2
# def function_2(x):
#     return x[0]**2 + x[1]**2
#
# print(fc.numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(fc.numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(fc.numerical_gradient(function_2, np.array([3.0, 0.0])))

# # diff example - find the minimum gradient
# def function_2(x):
#     return x[0]**2 + x[1]**2
#
#
# init_x = np.array([-3.0, 4.0])
# print(fc.gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# Gradient simple network example
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 1]) # 임의 정답 레이블
print(net.loss(x, t))

f = lambda w: net.loss(x, t)
dW = fc.numerical_gradient(f, net.W)
print(dW)

