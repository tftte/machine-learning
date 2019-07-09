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

# 范数
print("=" * 20)
print("=" * 20)
a = tf.ones([2, 2])
print(tf.norm(a))
print(tf.sqrt(tf.reduce_sum(tf.square(a))))

print("*****每个维度的二范数*****")
print(tf.norm(a, ord=2, axis=1))

print("*********一范数*********")
print(tf.norm(a, ord=1))

print("**reduce_min/max/mean**")
b = tf.random.normal([4, 10])
print(tf.reduce_min(b), tf.reduce_max(b), tf.reduce_mean(b))

x = tf.constant([[1., 1.], [2., 2.]])
print(tf.reduce_mean(x))
print(tf.reduce_mean(x, 0))
print(tf.reduce_mean(x, 1))

print("********arg********")
print(tf.argmax(b).shape)
print(tf.argmax(b, axis=1))

print("*******equal*******")
c1 = tf.constant([1, 2, 2, 3, 4])
c2 = tf.range(5)
res = tf.equal(c1, c2)
print(res)
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32)))

print("*******unique******")
d = tf.constant([4, 2, 2, 4, 3])
d1, d2 = tf.unique(d)
print(d1, d2)
