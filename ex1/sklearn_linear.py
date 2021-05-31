from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn是scikit-learn的简称，是一个基于Python的第三方模块。
# sklearn库集成了一些常用的机器学习方法，在进行机器学习任务时，并不需要实现算法，只需要简单的调用sklearn库中提供的模块就能完成大多数的机器学习任务。
# sklearn库是在Numpy、Scipy和matplotlib的基础上开发而成的，因此在sklearn的安装前，需要先安装这些依赖库。


path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

X = np.mat(X.values)
y = np.mat(y.values)
model = linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)
# X[:, 1]是在第一个维度里选取所有的元素，然后在第二个维度里选取第一个元素，返回仍是一个二维数据
# 而A1的作用是将任意二维数据平铺成一维
f = model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
