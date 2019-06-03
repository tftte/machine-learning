#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import random

from mxnet import nd, autograd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(loc=0, scale=1, shape=(num_examples, num_inputs))

labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
labels += nd.random.normal(loc=0, scale=0.01, shape=labels.shape)


# 读取数据
def data_iter(bath_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, bath_size):
        j = nd.array(indices[i: min(i + bath_size, num_examples)])
        yield features.take(j), labels.take(j)


# 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    # reshape防止一个是一行一个是一列导致自动广播
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


if __name__ == '__main__':
    batch_size = 10
    lr = 0.03
    num_epoch = 3
    net = linreg
    loss = squared_loss

    # 初始化模型参数
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()

    for epoch in range(num_epoch):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()
            sgd([w, b], lr, batch_size)

        train_l = loss(net(features, w, b), labels)
        print('epoch: {}, loss: {}'.format(epoch + 1, train_l.mean().asnumpy()))

    print('true_W: {}, w: {}'.format(true_w, w))
    print('true_b: {}, b: {}'.format(true_b, b))
