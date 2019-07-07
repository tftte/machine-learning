#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

a = tf.ones([2, 2])
b = tf.fill([2, 2], 2.)
print(a + b, a - b, a * b, a / b)
print("====================")
print(a // b, a % b)
print("====================")
print(tf.math.log(a))
print(tf.exp(a))

# tf只提供log以e为底的api，要计算其他的需要做个除法
print("====================")
log2_8 = tf.math.log(8.) / tf.math.log(2.)
log10_100 = tf.math.log(100.) / tf.math.log(10.)
print(log2_8, log10_100)

# pow sqrt
print("====================")
print(tf.pow(b, 3))
print(b ** 3)
print(tf.sqrt(b))

# dot
print("====================")
print(a @ b)
print(tf.matmul(a, b))

# broadcasting
print("====================")
x = tf.ones([4, 2])
w = tf.ones([2, 1])
b = tf.constant(0.1)
print(x @ w + b)
