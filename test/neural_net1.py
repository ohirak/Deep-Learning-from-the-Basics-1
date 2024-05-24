"""手書き数字認識テスト
"""

import os, sys
sys.path.append(r"..\deep_learning")

import numpy as np
import pickle

from utility import math_util as mt

from datasets.mnist import load_mnist

def get_data():
    # MNISTデータセットの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(r"datasets\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    Z1 = mt.sigmoid(np.dot(x, network['W1']) + network['b1'])
    Z2 = mt.sigmoid(np.dot(Z1, network['W2']) + network['b2'])
    Y = mt.softmax(np.dot(Z2, network['W3']) + network['b3'])
    return Y

if __name__ == "__main__":
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    batch_size = 100

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p==t[i:i+batch_size])

    print("accuracy:\t", accuracy_cnt / len(x))