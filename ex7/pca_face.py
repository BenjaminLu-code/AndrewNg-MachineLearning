import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = 'data/ex7faces.mat'
data = loadmat(path)
X = data['X']
X = (X - X.mean(axis=0)) / X.std(axis=0)  # (5000, 1024)


def pca(X):
    covariance = (X.T @ X) / X.shape[0]
    U, S, V = np.linalg.svd(covariance)
    return U, S, V


U, S, V = pca(X)
print(S.shape)
count = 0
for i in np.arange(S.shape[0])+1:
    count = count +1
    if (np.sum(S[0:i]) / np.sum(S) - 0.99) > 0.000001:
        break
print(count)
print(U)
Z = U[:, 0:100]
# 这里要注意，返回的U中的每一列是一个特征向量，而不是一行
X_recover = (X @ Z) @ Z.T
plt.imshow(X_recover[0, :].reshape(32, 32).T, cmap='binary')
plt.show()
