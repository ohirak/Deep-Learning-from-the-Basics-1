
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
    c = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - c) # オーバーフロー対策のためx^exp_maxで割る
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x

def sum_squared_error(y, t):
    """二乗和誤差

    Args:
        y (np.array): 出力データ
        t (np.array): 訓練データ
    """
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    """交差エントロピー誤差

    Args:
        y (np.array): 出力データ
        t (np.array): 訓練データ
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # NOTE: t * np.log()の形にしないのは、tがワンホット表現でない場合もあるから
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_diff(f, x):
    """数値微分

    Args:
        f (_type_): _description_
        x (_type_): _description_
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x):
    """勾配

    Args:
        f (_type_): _description_
        x (_type_): _description_
    """
    grad = np.zeros_like(x)
    h = 1e-4
    iter = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not iter.finished:
        idx = iter.multi_index
        tmp_x = x[idx]
        x[idx] = tmp_x + h
        fx1 = f(x)
        x[idx] = tmp_x - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_x
        iter.iternext()
    return grad

def gradient_descent(f, init_x, step_num=100, lr=0.01):
    """勾配降下法

    Args:
        f (_type_): _description_
        init_x (_type_): _description_
        step_num (int, optional): _description_. Defaults to 100.
        lr (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    x = init_x
    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)
    return x
