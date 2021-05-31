import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = 'data/ex7data1.mat'
data = loadmat(path)
X = data['X']
X = (X - X.mean(axis=0)) / X.std(axis=0)
plt.scatter(X[:, 0], X[:, 1], c='r')

def pca(X):
    covariance = (X.T @ X) / X.shape[0]
    U, S, V = np.linalg.svd(covariance)
    return U, S, V


U, S, V = pca(X)
print(X.shape)
print(U[0:1, :].shape)
projection = (X @ U[0:1, :].T) * U[0:1, :]
plt.scatter(projection[:, 0], projection[:, 1], c='b')
plt.show()

# 在归一化之后，所有的数据都用归一化