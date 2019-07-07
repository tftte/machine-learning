#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf

a0 = tf.ones([1, 5, 5, 3])
print(a0[0, 0])
print(a0[0, 0, 0])
print(a0[0].shape)
print(a0[0, 0].shape)

a1 = tf.range(10)
# 下标-1到最后 shape=1
print(a1[-1:])
# 下标0到-1 左闭右开 shape=9
print(a1[:-1])

a2 = tf.random.normal([2, 4, 18, 18, 3])
print(a2[0, ...])
print(a2[..., 0])
print(a2[0, 0, ..., 2])

# 花式索引
a3 = tf.random.normal([4, 5, 8])
print(a3.shape)
print(tf.gather(a3, axis=0, indices=[2, 3]).shape)
print(a3[2:4].shape)
print(tf.gather(a3, axis=0, indices=[3, 1, 0]).shape)

# 多个gather联合使用
a31 = tf.gather(a3, axis=0, indices=[3, 0, 1])
a32 = tf.gather(a31, axis=1, indices=[2, 1, 3])
print(a32.shape)

# gather_nd
a4 = tf.gather_nd(a3, [0])
print(a4.shape)

a5 = a3[0]
print(a5.shape)

# 两个样本
a41 = tf.gather_nd(a3, [[0, 0], [1, 1]])
print(a41.shape)

# boolean_mask
a6 = tf.boolean_mask(a3, [True, True, False, False], axis=0)
print(a6.shape)

# mask对应的[2, 3]
a7 = tf.ones([2, 3, 4])
print(tf.boolean_mask(a7, mask=[[True, False, False],
                                [False, True, True]]))
