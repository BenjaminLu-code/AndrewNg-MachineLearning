import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])
data1 = data.copy()
# pandas的数据"="操作是给数据起别名，要将值复制一份，用copy
data['Zeros'] = 1.0
degree = 6
for i in range(1, degree+1):
    for j in range(0, i+1):
        data['F'+str(i-j)+str(j)] = np.power(data['Test 1'], i-j)*np.power(data['Test 2'], j)
# 对于pandas的数据进行次方操作竟然可以用numpy中的power!!!
data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)
col = data.shape[1]
X = data.loc[:, 'Zeros':'F06']
y = data.loc[:, 'Accepted']
# 对于DataFrame来说，'Zeros':'F06'是可以取到'F06'。
# 原因在于DataFrame进行的是标签索引，如果要取一堆列且包含最后一列的话，要是冒号不包含，最后一列永远没有办法一起取出来。
X = np.array(X)
y = np.array(y)
lam = 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costfun(theta, X, y, lam):
    theta1 = theta.copy()
    m1 = X@theta1
    m2 = sigmoid(m1)
    m3 = y*np.log(m2)+(1-y)*np.log(1-m2)
    m4 = -np.sum(m3)/y.shape
    theta1[0] = 0
    m5 = np.sum(np.power(theta1[1:theta1.shape[0]], 2))
# 这里的theta1因为是np类型，所以在用冒号时，取到的是1至，theta1.shape[0]-1号元素（最后一个）
    m6 = m5*lam/(2*y.shape[0])
    return m4+m6


def gradient(theta, X, y, lam):
    theta1 = theta.copy()
    m1 = np.matmul(X, theta1)
    m2 = sigmoid(m1)
    m3 = m2 - y
    m4 = np.matmul(m3.T, X)
    theta1[0] = 0
    m5 = (m4+lam*theta1)/y.shape[0]
    return m5


# theta = np.ones(X.shape[1])
# theta = np.random.rand(X.shape[1])
theta = np.ones(X.shape[1])
result = opt.fmin_tnc(func=costfun, x0=theta, fprime=gradient, args=(X, y, lam))
# 如果这里出错，大概率是矩阵相乘的维度问题


def predict(X, theta):
    m1 = sigmoid(np.matmul(X, theta))
    m2 = np.zeros(m1.shape)
    for i in np.arange(m1.shape[0]):
        if m1[i] >= 0.5:
            m2[i] = 1
    return m2


prediction = predict(X, result[0])
results = np.zeros(y.shape)
for i in np.arange(y.shape[0]):
    if prediction[i] == y[i]:
        results[i] = 1
rate = np.sum(results)/y.shape
print('预测正确率为', rate)

X1 = np.arange(-0.8, 1.2, 0.01)
X2 = np.arange(-0.8, 1.2, 0.01)
cordinates = [(x1, x2) for x1 in X1 for x2 in X2]
x1_cord, x2_cord = zip(*cordinates)
cord = pd.DataFrame({"x1": x1_cord, "x2": x2_cord})
cord['Zeros'] = 1.0
cord['Zeros'] = cord['Zeros']*result[0][0]
count = 1
for i in range(1, degree+1):
    for j in range(0, i+1):
        cord['F'+str(i-j)+str(j)] = np.power(cord['x1'], i-j)*np.power(cord['x2'], j)*result[0][count]
        count = count + 1
cord['sum'] = cord.sum(axis=1) - cord['x1'] - cord['x2']
# 上面这一段代码逻辑为，我要找到判断0还是1的边界，但是这里的边界是横纵坐标高达6次方的方程，就不能像前面线性一样取x，用优化的theta算出y
# 然后画图，这里需要在图上打满散点，将每个点的横纵坐标带入theta的方程，绝对值小于0.001的就是边界上的点，然后画出这些点。
positive = data1[data1['Accepted'].isin([1])]
negative = data1[data1['Accepted'].isin([0])]
plt.figure(figsize=(12, 8))
plt.scatter(positive['Test 1'], positive['Test 2'], s=50, c='r', marker='o', label='Accepted')
plt.scatter(negative['Test 1'], negative['Test 2'], s=50, c='b', marker='x', label='Not Accepted')
plt.scatter(cord[abs(cord['sum']) < 0.01]['x1'], cord[abs(cord['sum']) < 0.01]['x2'], s=25, c='y', label='Boundry')
plt.legend()
plt.xlabel('Test 1')
plt.ylabel('Test 2')
plt.show()
