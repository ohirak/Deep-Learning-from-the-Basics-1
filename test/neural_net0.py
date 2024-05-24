"""3層ニューラルネットワークの実装
"""

import numpy as np
import os
import sys
sys.path.append(r"..\deep_learning")
from utility import math_util as mt

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    """入力から出力への変換プロセス

    Args:
        network (dict): 各層の重み、バイアスの辞書
        x (np.array): 入力
    """
    Z1 = mt.sigmoid(np.dot(x, network['W1']) + network['b1'])
    Z2 = mt.sigmoid(np.dot(Z1, network['W2']) + network['b2'])
    Y = mt.identity(np.dot(Z2, network['W3']) + network['b3'])
    return Y

if __name__ == "__main__":
    network = init_network()
    X = np.array([1.0, 0.5])
    Y = forward(network, X)
    print(Y)

    
