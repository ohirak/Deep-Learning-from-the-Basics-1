"""勾配降下法で学習する2層ニューラルネット
"""

import os, sys
sys.path.append(r"..\deep_learning")

import numpy as np

from utility import math_util as mt
from datasets import mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        Z = mt.sigmoid(np.dot(x, self.params['W1']) + self.params['b1'])
        Y = mt.softmax(np.dot(Z, self.params['W2']) + self.params['b2'])
        return Y
    
    def loss(self, x, t):
        y = self.predict(x)
        return mt.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        Y = self.predict(x)
        Y = np.argmax(Y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(Y==t) / x.shape[0]
        return accuracy
    
    def numerical_gradient(self, x, t):
        f = lambda w: self.loss(x, t)
        grads = {}
        grads['W1'] = mt.numerical_gradient(f, self.params['W1'])
        grads['b1'] = mt.numerical_gradient(f, self.params['b1'])
        grads['W2'] = mt.numerical_gradient(f, self.params['W2'])
        grads['b2'] = mt.numerical_gradient(f, self.params['b2'])
        return grads

# ミニバッチ学習
(x_train, t_train), (x_test, t_test) = \
    mnist.load_mnist(normalize=True, one_hot_label=True)

lr = 0.1
step_num = 100
batch_size = 100
train_size = t_train.shape[0]
net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(step_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]
    grads = net.numerical_gradient(x, t)
    net.params['W1'] -= lr * grads['W1']
    net.params['b1'] -= lr * grads['b1']
    net.params['W2'] -= lr * grads['W2']
    net.params['b2'] -= lr * grads['b2']

    print("step", i, ":\t", net.loss(x, t))