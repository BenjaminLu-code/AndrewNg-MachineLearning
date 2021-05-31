import numpy as np
import pandas as pd
# pandas包含两种最主要数据结构：序列（Series）和数据框（DataFrame）。
# 对于这两个数据结构，有两个最基本的概念：轴（Axis）和标签（Label），
# 对于二维数据结构，轴是指行和列，轴标签是指行的索引和列的名称，存储轴标签的数据结构是Index结构。
import matplotlib.pyplot as plt


def computecost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * int(X.shape[0]))
# 这个函数是代价函数的定义式，第一行计算出每一个数据点的差平方，
# 然后在第二行将array对象中的元素全部相加，且除以两倍数据点个数


def gradientdescent(X, y, theta, alpha, iters):
    temp = np.copy(theta)
    parameters = int(theta.shape[1])
# ravel是扁平化函数，虽然ravel返回的对象在新的地址，
# 但是改变返回对象的值原对象值也会改变，parameters是theta参数个数
    cost = np.zeros(iters)
    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            term = np.matmul(error.T, X[:, j])
            # np中基本运算要熟悉
            a = (alpha / int(X.shape[0]) * term)
            temp[0, j] = theta[0, j] - a
            # 在前面的theta初值要设成0.0而不是0，整形的话0-0.1=0

        theta = np.copy(temp)
        cost[i] = computecost(X, y, theta)

    return theta, cost


path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# read_csv也可以导入txt文件
# print(data.head())
# print(data.describe())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()
data.insert(0, 'Ones', 1)
# 第一个参数是位置，第二个参数是列名，第三个参数的值
# set X (training data) and y (target variable)
cols = data.shape[1]
# 为1时shape返回一行有多少个元素，为0时shape返回一列有多少个元素
X = data.iloc[:, 0:cols-1]
# X是所有行，去掉最后一列
y = data.iloc[:, cols-1:cols]
# X是所有行，最后一列
'''import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(4, 4), index=list('0246'), columns=list('ABCD'))
print(df)
print(df.iloc[1, 1])
print(df.loc['4'])'''
# 示例中iloc函数返回的是第二行第二列的元素，loc返回的是标签为4的所有元素，即第三行所有元素
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.array([0.0, 0.0]))
alpha = 0.002
iters = 10000
# 设置迭代次数和学习率需要多尝试几次，我在这里设置的参数比参考代码里的参数找到的代价函数值要小
g, cost = gradientdescent(X, y, theta, alpha, iters)
print(g)
print(computecost(X, y, g))


x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(cost)
ax2.plot(x, f, 'r', label='Prediction')
ax2.scatter(data.Population, data.Profit, label='Traning Data')
ax2.legend(loc=2)
ax2.set_xlabel('Population')
ax2.set_ylabel('Profit')
ax2.set_title('Predicted Profit vs. Population Size')
plt.show()