import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data1 = data.copy()
data['Exam 1'] = (data['Exam 1'] - data['Exam 1'].mean()) / data['Exam 1'].std()
data['Exam 2'] = (data['Exam 2'] - data['Exam 2'].mean()) / data['Exam 2'].std()
# 放缩非常重要，要是不放缩，在求损失函数的时候，log里面值会大多都会是0或1
data.insert(0, 'Zeros', 1.0)
col = data.shape[1]
X = data.iloc[:, 0:col-1]
y = data.iloc[:, col-1]

'''plt.figure(figsize=(12, 8))
plt.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()'''


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#  这里为什么要写指数函数要用np，因为python原生没有指数函数，math和np里面有


def costfun(X, y, theta):
    m1 = np.matmul(X, theta)
    m2 = sigmoid(m1)
    m3 = y*np.log(m2)+(1-y)*np.log(1-m2)
    m4 = -np.sum(m3)/y.shape
    return m4
    
    
def find(X, y, theta, alpha, iters):
    cost = np.zeros(iters)
    print(y.shape)
    for i in np.arange(iters):
        m1 = np.matmul(X, theta)
        m2 = sigmoid(m1)
        m3 = m2 - y
        m4 = np.matmul(m3.T, X)
        theta = theta - alpha / y.shape[0] * m4
        cost[i] = costfun(X, y, theta)
    return theta, cost


X = np.array(X)
y = np.array(y)
#theta = np.zeros(col-1)
theta = np.array([0.0, 0.0, 0.0])
alpha = 0.1
iters = 10000
g, cost = find(X, y, theta, alpha, iters)
print(g)
print(cost[-1])
# plt.plot(cost)
# plt.show()

positive = data1[data1['Admitted'].isin([1])]
negative = data1[data1['Admitted'].isin([0])]
plt.figure(figsize=(12, 8))
plt.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
x1 = np.linspace(data1['Exam 1'].min(), data1['Exam 2'].max(), 100)
x2 = -(g[1]*data1['Exam 2'].std())/(g[2]*data1['Exam 1'].std())*(x1-data1['Exam 1'].mean())-g[0]/g[2]*data1['Exam 2'].std()+data1['Exam 2'].mean()
plt.plot(x1, x2)
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()


def predict(X, theta):
    m1 = sigmoid(np.matmul(X, theta))
    m2 = np.zeros(m1.shape)
    for i in np.arange(m1.shape[0]):
        if m1[i] >= 0.5:
            m2[i] = 1
    return m2


prediction = predict(X, theta)
results = np.zeros(y.shape)
for i in np.arange(y.shape[0]):
    if prediction[i] == y[i]:
        results[i] = 1
rate = np.sum(results)/y.shape
print('预测正确率为', rate)