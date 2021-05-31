import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

path1 = 'ex4data1.mat'
data = loadmat(path1)
X = data['X']  # (5000, 400)
y = data['y']  # (5000, 1)
path2 = 'ex4weights.mat'
theta = loadmat(path2)
theta1 = theta['Theta1']  # (25, 401)
theta2 = theta['Theta2']  # (10, 26)


def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


def deserialize(seq):
#     """into ndarray of (25, 401), (10, 26)"""
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


theta = serialize(theta1, theta2)
# 因为优化只能有一个对象，因此我们需要将theta1和theta2整成一个array
# 然后输进代价和梯度之中，由他们自己再变成theta1 和 theta2
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costfun(theta, X, y):
    theta1, theta2 = deserialize(theta)
    X = np.insert(X, 0, 1, axis=1)
    z2 = X @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    y1 = np.zeros_like(a3)
    # y[y == 10] = 0
    for i in np.arange(y.shape[0]):
        y1[i, y[i]-1] = 1
    # 原始训练是将最后一个标定为0分类
    cost = np.sum(-np.multiply(y1, np.log(a3)) - np.multiply(1-y1, np.log(1-a3))) / X.shape[0]
    return X, z2, a2, z3, a3, cost


def regularized_costfun(theta, X, y, l):
    theta1, theta2 = deserialize(theta)
    X, z2, a2, z3, a3, cost = costfun(theta, X, y)
    theta1 = np.delete(theta1, 0, axis=1)
    theta2 = np.delete(theta2, 0, axis=1)
    r_cost = (np.sum(np.power(theta1, 2)) + np.sum(np.power(theta2, 2))) * l / (2 * X.shape[0])
    return X, z2, a2, z3, a3, r_cost + cost


def gradient(theta, X, y):
    theta1, theta2 = deserialize(theta)
    X1, z2, a2, z3, a3, cost = costfun(theta, X, y)
    y1 = np.zeros_like(a3)
    for i in np.arange(y.shape[0]):
        y1[i, y[i] - 1] = 1
    delta3 = a3 - y1  # (5000, 10)
    delta2 = delta3 @ theta2 * a2 * (1 - a2)  # (5000, 26)
    D2 = delta3.T @ a2
    D1 = delta2[:, 1:delta2.shape[1]].T  @ X1
    D2 = D2 / X.shape[0]
    D1 = D1 / X.shape[0]
    return serialize(D1, D2)


def regularized_gradient(theta, X, y, l):
    theta1, theta2 = deserialize(theta)
    X1, z2, a2, z3, a3, cost = costfun(theta, X, y)
    y1 = np.zeros_like(a3)
    for i in np.arange(y.shape[0]):
        y1[i, y[i] - 1] = 1
    delta3 = a3 - y1  # (5000, 10)
    delta2 = delta3 @ theta2 * a2 * (1 - a2)  # (5000, 26)
    D2 = delta3.T @ a2
    D1 = delta2[:, 1:delta2.shape[1]].T @ X1
    D2 = D2 / X.shape[0]
    D1 = D1 / X.shape[0]
    theta2[:, 0] = 0
    theta1[:, 0] = 0
    D2 = D2 + l * theta2
    D1 = D1 + l * theta1
    return serialize(D1, D2)


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def nn_training(X, y):
    """regularized version
    the architecture is hard coded here... won't generalize
    """
    # init_theta = random_init(10285)  # 25*401 + 10*26
    init_theta = np.zeros(10285)
    res = opt.fmin_tnc(func=costfun, x0=init_theta, fprime=gradient, args=(X, y))
    '''
    res = opt.minimize(fun=costfun,
                       x0=init_theta,
                       args=(X, y),
                       method='TNC',
                       jac=gradient,
                       options={'maxiter': 400})
    '''
    print(type(res))
    return res



res = nn_training(X, y)#慢
print(res[0].shape)
'''
theta1, theta2 = deserialize(regularized_gradient(theta, X, y, 1))
print(theta1.shape)
print(theta2.shape)
'''
