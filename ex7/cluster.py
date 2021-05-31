import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = 'data/ex7data2.mat'
data = loadmat(path)
X = data['X']


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
        centroids[i, :] = np.sum(points, 0) / points.shape[0]
    return centroids



def run_K_means(X, centroids, max_iters):
    for i in np.arange(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, centroids.shape[0])
    return idx, centroids


init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
max_iters = 100
idx, centroids = run_K_means(X, init_centroids, max_iters)
print(idx, centroids)
for k in np.arange(centroids.shape[0]):
    idx1 = idx == k
    points = X[idx1.ravel(), :]
    if k == 0:
        c = 'r'
    elif k == 1:
        c = 'y'
    else:
        c = 'b'
    plt.scatter(points[:, 0], points[:, 1], c=c, s=5)
    plt.scatter(centroids[k, 0], centroids[k, 1], c=c, s=50)
plt.show()
