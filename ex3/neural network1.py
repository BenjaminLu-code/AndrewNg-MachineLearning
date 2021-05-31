import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report

data = loadmat('ex3data1.mat')
X = data['X'].copy()
y = data['y'].copy()
X = X.T
X = np.insert(X, 0, 1, axis=0)
y = y.T

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costfun(theta, X, y, i):
    theta = theta.reshape((1, theta.shape[0]))
    m1 = np.matmul(theta, X)
    m2 = sigmoid(m1)
    m3 = np.zeros((theta.shape[0], X.shape[1]))
    if i == 0:
        m3[0, :] = y == 10
    else:
        m3[0, :] = y == i
    m4 = m3 * np.log(m2) + (1 - m3) * np.log(1-m2)
    m5 = -np.sum(m4, axis=1) / m2.shape[1]
    return m5


def gradient(theta, X, y, i):
    theta = theta.reshape((1, theta.shape[0]))
    m1 = np.matmul(theta, X)
    m2 = sigmoid(m1)
    m3 = np.zeros((theta.shape[0], X.shape[1]))
    if i == 0:
        m3[0, :] = y == 10
    else:
        m3[0, :] = y == i
    m4 = m2 - m3
    m5 = np.matmul(m4, X.T) / X.shape[1]
    return m5


clanum = 10
# theta = np.zeros((clanum, X.shape[0]))
theta = np.random.random((clanum, X.shape[0]))
# theta = np.full((clanum, X.shape[0]), 0.1)
# 这里为什么当初始的theta值较大时，最后的出来的预测向量没有办法判断是哪个结果
print(X.shape)
print(y.shape)
print(theta.shape)
#costfun(theta, X, y)
#gradient(theta, X, y)
for i in np.arange(clanum):
    theta_in = theta[i:i+1, :]
    fmin = opt.fmin_tnc(func=costfun, x0=theta_in, fprime=gradient, args=(X, y, i))
    theta[i, :] = fmin[0].copy()
print(theta)

def predict(X, theta):
    m1 = np.matmul(theta, X)
    m2 = sigmoid(m1)
    '''
    for i in np.arange(m2.shape[1]):
        m2[:, i] = abs(m2[:, i] - m2[:, i].max()) < 0.00000001
        m4 = np.where(m2[:, i] == m2[:, i].max())
        m3 = len(m4[0])
        if m3 != 1:
            print('无法预测第'+str(i)+'例子')
    '''
    m3 = np.argmax(m2, axis=0)
    return m3


prediction = predict(X, theta)
print(prediction.shape)
print(y.shape)
y1 = y[0].copy()
y1[y1 == 10] = 0

'''
results = np.zeros((1, X.shape[1]))
for i in np.arange(X.shape[1]):
    if y[0, i] == 10:
        if prediction[0, i] == 1:
            results[0, i] = 1
    else:
        if prediction[y[0, i], i] == 1:
            results[0, i] = 1
rate = np.sum(results)/results.shape[1]
print('预测正确率为', rate)
'''
print(classification_report(y1, prediction))


# 用两种方法处理梯度下降得到的结果，在初始theta取随机数的时候都有预测结果的矢量有多个分量相同
# 宗量z过大导致各个分量皆为1，所以初值敏感，还是要取theta = 0
# 题目提供的python程序也有这个问题，应该是内嵌优化函数本身的问题