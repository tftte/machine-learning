#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

a = tf.random.shuffle(tf.range(5))
print(a)
print(tf.sort(a, direction='DESCENDING'))
idx = tf.argsort(a, direction='DESCENDING')
print(idx)

print(tf.gather(a, idx))


b = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
print(b)
print(tf.sort(b))
print(tf.argsort(b))

print("*******top_k********")
res = tf.math.top_k(b, 2)
print(res.indices)
print(res.values)
