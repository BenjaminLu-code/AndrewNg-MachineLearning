import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def computecost(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner)/(2 * y.shape[0])


def gradientdescent(X, y, theta, alpha, iters):
    temp = np.copy(theta)
    cost = np.zeros(iters)
    parameters = theta.shape[1]
    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            term = error.T * X[:, j]
            a = alpha * term / int(X.shape[0])
           # print(a)
            temp[0, j] = theta[0, j] - a
        theta = np.copy(temp)
        cost[i] = computecost(X, y, theta)
    return theta, cost


data = pd.read_csv('ex1data2.txt', names=['size', 'bedroomnumber', 'prize'])
data = (data - data.mean()) / data.std()
data.insert(0, 'ones', 1)
col = data.shape[1]
X = data.iloc[:, 0:col-1]
y = data.iloc[:, col-1]
# data.iloc[:, col-1]返回的是最后一列的一个Series，而data.iloc[:, col-1:col]返回最后一列的DataFrame
# 这个原因是前者的第二个坐标是一个值，因此维度降低，而后者第二个坐标形式为两个值（尽管第二个值访问不到元素）
# 因此维度不会降低，返回仍是DataFrame。col-1:col+1也不会报错，返回是同样结果。这个特性要用在将DataFrame转换成np矩阵时候
# 前者转换成的维度是(1, 47)，后者维度是(47, 1)

X = np.mat(X)
y = np.mat(y)
alpha = 0.05
iters = 100
theta = np.mat([0.0, 0.0, 8.0])
g, cost = gradientdescent(X, y, theta, alpha, iters)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

