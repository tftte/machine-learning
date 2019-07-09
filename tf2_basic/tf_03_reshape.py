#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

a = tf.random.normal([4, 28, 28, 3])
print(a.shape, a.ndim)

a1 = tf.reshape(a, [4, 28 * 28, 3])
print(a.shape)
print(a1.shape)

a11 = tf.reshape(a, [4, -1, 3])
print(a11.shape)

# 转置
print("********transpose********")
a2 = tf.random.normal([1, 2, 3, 4])
a3 = tf.transpose(a2)
print(a2.shape)
print(a3.shape)
a4 = tf.transpose(a2, perm=[0, 2, 3, 1])
print(a4.shape)

# 增加维度(索引前面)
print("********增加维度********")
b = tf.random.normal([4, 35, 8])
print(tf.expand_dims(b, axis=0).shape)
print(b.shape)

# 减少维度(去掉shape=1的axis)
c = tf.zeros([1, 2, 1, 3])
print(tf.squeeze(c).shape)
print(c.shape)

# broadcasting
print("********broadcasting********")
d = tf.ones([3, 4])
d1 = tf.broadcast_to(d, [2, 3, 4])
print(d1.shape)

d2 = tf.expand_dims(d, axis=0)
print(d2.shape)
d2 = tf.tile(d2, [2, 1, 1])
print(d2.shape)
