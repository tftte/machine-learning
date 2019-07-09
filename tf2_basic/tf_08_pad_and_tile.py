#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

# pad
a = tf.reshape(tf.range(9), shape=[3, 3])
print(a)
print(tf.pad(a, [[0, 1], [1, 1]]))

# image padding
b = tf.random.normal([4, 28, 28, 3])
c = tf.pad(b, [[0, 0], [2, 2], [2, 2], [0, 0]])
print(b.shape, c.shape)

# tile
print(tf.tile(a, [1, 2]))  # 行复制1倍，列复制2倍
print(tf.tile(a, [2, 1]))
