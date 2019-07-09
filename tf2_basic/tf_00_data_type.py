#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import numpy as np
import tensorflow as tf

# TensorFlow数据类型
a1 = tf.constant(1)
print("a1: ", a1)

a2 = tf.constant(1.)
print("a2: ", a2)

a3 = tf.constant(2.2, dtype=tf.double)
print("a3: ", a3)

a4 = tf.constant([True, False])
print("a4: ", a4)

a5 = tf.constant("Hello TensorFlow!")
print("a5: ", a5)

# device
with tf.device("cpu"):
    a6 = tf.constant(1)

with tf.device("cpu:0"):
    a7 = tf.range(4)

print("a6.device: ", a6.device)
print("a6: ", a6)

# aa = a6.gpu()

print("a7.device: ", a7.device)
print("a7: ", a7)

# tensor -> np
a8 = a7.numpy()
# a8 = a7
print("a8: ", a8,
      " type: ", type(a8),
      " shape: ", a8.shape,
      " dim: ", a8.ndim)
print("tf.rank(a8): ", tf.rank(a8))

print("tf.is_tensor(a8)", tf.is_tensor(a8))
print("a7.dtype == tf.int32: ", a7.dtype == tf.int32)

a9 = np.arange(5)
print("a9: ", a9.dtype)

# np -> tensor
a10 = tf.convert_to_tensor(a9)
print("a10: ", a10)

a10 = tf.convert_to_tensor(a9, dtype=tf.int32)
print("a10: ", a10)

# 类型转换
a11 = tf.constant([1.1, 2.2, 9.9])
a12 = tf.cast(a11, dtype=tf.int32)
print("a12: ", a12)

b0 = tf.constant([0, 1])
b1 = tf.cast(b0, dtype=tf.bool)
b2 = tf.cast(b1, dtype=tf.int32)
print("b0: ", b0)
print("b1: ", b1)
print("b2: ", b2)

# variable
b3 = tf.range(5)
b4 = tf.Variable(b3)
print("b3: ", b3)
print("b4: ", b4)
print("b4.dtype: ", b4.dtype,
      " b4.name: ", b4.name,
      " b4.trainable: ", b4.trainable)

b5 = tf.ones([])
b6 = b5.numpy()
b7 = int(b5)
b8 = float(b5)
print("b5: ", b5,
      " b6: ", b6,
      " b7: ", b7,
      " b8 ", b8)
