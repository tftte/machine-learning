#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf
from tensorflow import keras

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x.shape, y.shape)
print(x_test.shape, y_test.shape)
print(x.min(), x.max(), x.mean())

print(y[:5])

y_onehot = tf.one_hot(y, depth=10)
print(y_onehot.shape)
print(y_onehot[:4])

db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db = db.shuffle(1000)
print(next(iter(db))[0].shape)
print(next(iter(db))[1].shape)


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


db2 = db.map(preprocess)
res = next(iter(db2))
print(res[0].shape, res[1].shape)
print(res[1])

db3 = db2.batch(32)
res = next(iter(db3))
print(res[0].shape, res[1].shape)