
import numpy as np

def step_function(x):
    """step関数

    Args:
        x (np.array): _description_
    """
    bool_x  = (x > 0)
    int_x   = bool_x.astype(int)
    return int_x

def sigmoid(x):
    """シグモイド関数

    Args:
        x (np.array): _description_ 
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Relu関数

    Args:
        x (np.array): _description_
    """
    return np.maximum(0, x)

def identity(x):
    """恒等関数

    Args:
        x (_type): _description_
    """
    return x

def softmax(x):
    """ソフトマックス関数

    Args:
        x (np.array): _description_
    """
    c = np.max(x)
    exp_x = np.exp(x - c) # オーバーフロー対策のためx^exp_maxで割る
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
    

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

import matplotlib.pyplot as plt

plt.plot(x,y)
plt.ylim(-0.1, 5.1)
plt.show()

a = np.array([0.3, 2.9, 4.0])
y=softmax(a)
print(y)
