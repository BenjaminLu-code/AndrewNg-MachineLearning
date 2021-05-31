import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import math

path = 'data/bird_small.mat'
data = loadmat(path)
A = data['A']
As = A.shape
A = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
A = A / 255


def find_closest_centroids(X, centroids):
    k = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros(m)

    for i in np.arange(m):
        min_dist = 100000000
        for j in np.arange(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


def compute_centroids(X, idx, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in np.arange(k):
        idx1 = idx == i
        points = X[idx1.ravel(), :]
        if points.shape[0] != 0:
            centroids[i, :] = np.sum(points, 0) / points.shape[0]
    return centroids


def run_K_means(X, centroids, max_iters):
    for i in np.arange(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, centroids.shape[0])
    return idx, centroids


init_centroids = np.random.rand(16, 3)
max_iters = 5
idx, centroids = run_K_means(A, init_centroids, max_iters)
print(idx.shape)
for i in range(idx.shape[0]):
    A[i, :] = centroids[int(idx[i]), :]
A = A.reshape(As)
plt.imshow((np.floor(A*255)).astype(int))
plt.show()
# 常规照片，几千万像素，根本搞不动
