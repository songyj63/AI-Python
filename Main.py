import numpy as np
import sys, os
from dataset.mnist import load_mnist
from PIL import Image
import NeuralNet_Mnist_Eaxmple as nme


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

# MNIST example, predict & accuracy using batch
x, t = nme.get_data()
network = nme.init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = nme.predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))