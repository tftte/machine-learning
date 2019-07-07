#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import numpy as np
import tensorflow as tf

a0 = tf.convert_to_tensor(np.ones([2, 3]))
print("a0: ", a0)

a1 = tf.convert_to_tensor([1, 2])
print("a1: ", a1)

a2 = tf.zeros([])
print("a2: ", a2)

a3 = tf.zeros([1])
print("a3: ", a3)

a4 = tf.zeros([2, 3, 3])
print("a4: ", a4)

a5 = tf.ones_like(a4)
a6 = tf.zeros_like(a5)
a7 = tf.zeros(a5.shape)
print("a5: ", a5)
print("a6: ", a6)
print("a7: ", a7)

a8 = tf.ones(1)
a9 = tf.ones([])
print("a8: ", a8)
print("a9: ", a9)

a10 = tf.fill([2, 2], 3)
print("a10: ", a10)

# 正态分布
b0 = tf.random.normal([2, 2])
print("b0: ", b0)

b1 = tf.random.normal([2, 2], mean=1, stddev=1)
print("b1: ", b1)

# 截断的正态分布
b2 = tf.random.truncated_normal([2, 2])
print("b2: ", b2)

# 均匀分布
b3 = tf.random.uniform([2, 2], minval=0, maxval=1)
print(b3)

b4 = tf.random.uniform([2, 2])
print(b4)

index = tf.range(10)
index = tf.random.shuffle(index)
print(index)
