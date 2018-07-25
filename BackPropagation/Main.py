import BackPropagation.Layer_Naive as ln
import numpy as np


# # apple example with MulLayer
# apple = 100
# apple_num = 2
# tax = 1.1
#
# # 계층들
# mul_apple_layer = ln.MulLayer()
# mul_tax_layer = ln.MulLayer()
#
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_tax_layer.forward(apple_price, tax)
#
# print(price)
#
# # 역전파
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
#
# print(dapple, dapple_num, dtax)


# # apple and orange example
# apple = 100
# apple_num = 2
# orange = 150
# orange_num = 3
# tax = 1.1
#
# # 계층들
# mul_apple_layer = ln.MulLayer()
# mul_orange_layer = ln.MulLayer()
# add_apple_orange_layer = ln.AddLayer()
# mul_tax_layer = ln.MulLayer()
#
# # 순전파
# apple_price = mul_apple_layer.forward(apple, apple_num)
# orange_price = mul_orange_layer.forward(orange, orange_num)
# all_price = add_apple_orange_layer.forward(apple_price, orange_price)
# price = mul_tax_layer.forward(all_price, tax)
#
# # 역전파
# dprice = 1
# dall_price, dtax = mul_tax_layer.backward(dprice)
# dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
#
# print(price)
# print(dapple_num, dapple, dorange, dorange_num, dtax)

# # 기울기 확인 오차역전법 & 수치 미분
# from dataset.mnist import load_mnist
# from BackPropagation.TwoLayerNet import TwoLayerNet
#
# # 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#
# x_batch = x_train[:3]
# t_batch = t_train[:3]
#
# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# grad_backprop = network.gradient(x_batch, t_batch)
#
# # 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
# for key in grad_numerical.keys():
#     diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
#     print(key+":"+str(diff))

# 오차역전파법을 사용한 학습 구현

import numpy as np
from dataset.mnist import load_mnist
from BackPropagation.TwoLayerNet import TwoLayerNet
import matplotlib.pylab as plt

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기를 구한다
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc: " + str(train_acc) + " " + str(test_acc))


# 그래프 그리기
plt.figure(1)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(range(iters_num), train_loss_list)

plt.figure(2)
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.show()