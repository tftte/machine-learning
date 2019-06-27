#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'hh'

from mxnet import nd, autograd


def f(x):
    return x * 2


if __name__ == '__main__':
    x = nd.arange(12).reshape(3, 4)
    x.attach_grad()
    with autograd.record():
        y = f(x)

    y.backward()
    print(x.grad)
