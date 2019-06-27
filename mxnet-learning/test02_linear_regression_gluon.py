#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

from mxnet import nd, autograd
# from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(loc=0, scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, nd.array(true_w).T) + true_b
labels += nd.random.normal(loc=0, scale=0.01, shape=labels.shape)

# plt.plot(features[:, 1].asnumpy(), labels.asnumpy(), 'o')
# plt.show()

if __name__ == '__main__':

    # 读取数据
    batch_size = 10
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential()
    net.add(nn.Dense(1))  # 输出个数为1, 自动推断每一层的输入个数

    # 初始化模型参数
    net.initialize(init.Normal())  # 默认sigma=0.01

    # 定义损失函数
    loss = gloss.L2Loss()

    # 定义优化算法
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)

        l = loss(net(features), labels)
        print('epoch: {}, loss: {}'.format(epoch+1, l.mean().asnumpy()))

    dense = net[0]
    print('true_w: {}, w: {}'.format(true_w, dense.weight.data()))
    print('true_b: {}, b: {}'.format(true_b, dense.bias.data()))

