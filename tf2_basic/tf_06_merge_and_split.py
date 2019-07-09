#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
a1 = tf.ones([2, 35, 8])

# 不增加维度
c = tf.concat([a1, b], axis=0)
print(c.shape)

# 增加维度
c1 = tf.stack([a, b], axis=0)
print(c1.shape)

d = tf.stack([a, b], axis=3)
print(d.shape)

print("====================")
# unstack
aa, bb = tf.unstack(c1, axis=0)
print(aa.shape, bb.shape)

res = tf.unstack(d, axis=0)
print(type(res), len(res), res[0].shape)

print("====================")
res = tf.split(d, axis=2, num_or_size_splits=2)
print(res[0].shape, res[1].shape)

print("====================")
print(d.shape)
res = tf.split(d, axis=2, num_or_size_splits=[2, 4, 2])
print(res[0].shape, res[1].shape, res[2].shape)
