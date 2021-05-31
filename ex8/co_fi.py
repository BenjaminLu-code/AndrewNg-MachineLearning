import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

path = 'data/ex8_movies.mat'
data = loadmat(path)
Y = data['Y']  # (1682, 943)
R = data['R']  # (1682, 943)
path = 'data/ex8_movieParams.mat'
data = loadmat(path)
X = data['X']  # (1682, 10)
theta = data['Theta']  # (943, 10)


def serialize(X, theta):
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_feature):
    return param[:n_movie*n_feature].reshape((n_movie, n_feature)), param[n_movie*n_feature:].reshape((n_user, n_feature))


def cofiCostFunc(param, Y, R, n_feature):
    X, theta = deserialize(param, Y.shape[0], Y.shape[1], n_feature)
    rating = X @ theta.T
    rating[R == 0] = 0
    return np.sum(np.power(rating-Y, 2)) / 2


def cofigradient(param, Y, R, n_feature):
    X, theta = deserialize(param, Y.shape[0], Y.shape[1], n_feature)
    X_gradient = ((X @ theta.T - y) * R) @ theta
    theta_gradient = ((X @ theta.T - y) * R).T @ X
    return np.concatenate(X_gradient.ravel(), theta_gradient.ravel())


def re_cofiCostFunc(param, Y, R, n_feature, l):
    p1 = cofiCostFunc(param, Y, R, n_feature)
    return p1 + np.sum(np.power(param, 2)) * l / 2


def re_cofigradient(param, Y, R, n_feature, l):
    p1 = cofigradient(param, Y, R, n_feature)
    return p1 + l * param


param = serialize(X, theta)
print(re_cofiCostFunc(param, Y, R, X.shape[1], 1))

