#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

import tensorflow as tf
from tensorflow import keras


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int32)
    return x, y


def mnist_dataset():
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.map(prepare_mnist_features_and_labels)
    ds_test = ds_test.shuffle(10000).batch(100)
    return ds, ds_test



