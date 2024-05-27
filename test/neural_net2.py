"""勾配降下法を用いたニューラルネット
"""

import os, sys
sys.path.append(r"..\deep_learning")

import numpy as np

from utility import math_util as mt

class SimpleNet:
    def __init__(self):
        # ガウス分布で初期化
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        Z = np.dot(x, self.W)
        return Z
    
    def loss(self, x, t):
        Y = mt.softmax(self.predict(x))
        loss = mt.cross_entropy_error(Y, t)
        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
net = SimpleNet()
f = lambda w: net.loss(x, t)
dW = mt.numerical_gradient(f, net.W)
print(dW)

net = SimpleNet()
net.W = [[0.47355232, 0.9977393, 0.84668094],
         [0.85557411, 0.03563661, 0.69422093]]
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [ 1.05414809 0.63071653 1.1328074]
print(np.argmax(p)) # 最大値のインデックス
#2
t = np.array([0, 0, 1]) # 正解ラベル
print(net.loss(x, t))
# 0.92806853663411326