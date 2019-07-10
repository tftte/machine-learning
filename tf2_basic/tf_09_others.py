#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf
from matplotlib import pyplot as plt

a = tf.range(10)

print(tf.clip_by_value(a, 2, 8))

a = a - 5

print(tf.nn.relu(a))
print(tf.maximum(a, 0))

b = tf.random.normal([3, 3])
mask = b > 0
indices = tf.where(mask)
print(indices)
print(tf.gather_nd(b, indices))

c = tf.ones([3, 3])
d = tf.zeros([3, 3])
print(tf.where(mask, c, d))

print("********meshgrid********")
x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)
px, py = tf.meshgrid(x, y)
print(px.shape, py.shape)
points = tf.stack([px, py], axis=2)
z = tf.math.sin(points[..., 0]) + tf.math.sin(points[..., 1])
print(z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(px, py, z)
plt.colorbar()
plt.show()
