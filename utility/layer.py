from abc import ABCMeta, abstractmethod

import os, sys
sys.path.append(r"..\deep_learning")

import numpy as np

from utility import math_util as mt

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = mt.sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = mt.softmax(x)
        loss = mt.cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx