import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

path = 'data/ex8data1.mat'
data = loadmat(path)
X = data['X']  # (307, 2)
Xval = data['Xval']  # (307, 2)
yval = data['yval']  # (307, 1)


def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma


def selectThreshold(Xval, yval, mu, sigma):
    mo_1 = stats.norm(mu[0], sigma[0])
    pro_1 = mo_1.pdf(X[:, 0])
    mo_2 = stats.norm(mu[1], sigma[1])
    pro_2 = mo_2.pdf(X[:, 1])
    pro = pro_1 * pro_2
    best_epsilon = 0
    best_F1 = 0
    step = (pro.max() - pro.min()) / 1000
    for epsilon in np.arange(pro.min(), pro.max(), step):
        prediction = np.zeros(yval.shape)
        prediction[pro < epsilon] = 1
        true_positive = np.sum(np.logical_and(prediction == 1, yval == 1)).astype(float)
        false_positive = np.sum(np.logical_and(prediction == 1, yval == 0)).astype(float)
        true_negative = np.sum(np.logical_and(prediction == 0, yval == 0)).astype(float)
        false_negative = np.sum(np.logical_and(prediction == 0, yval == 1)).astype(float)
        prec = true_positive / (true_positive + false_positive)
        rec = true_positive / (true_positive + false_negative)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > best_F1:
            best_epsilon = epsilon
            best_F1 = F1
    return best_epsilon, best_F1


def prediction(X, mu, sigma, epsilon):
    mo_1 = stats.norm(mu[0], sigma[0])
    pro_1 = mo_1.pdf(X[:, 0])
    mo_2 = stats.norm(mu[1], sigma[1])
    pro_2 = mo_2.pdf(X[:, 1])
    pro = pro_1 * pro_2
    positive = X[pro < epsilon, :]
    negative = X[pro > epsilon, :]
    plt.scatter(positive[:, 0], positive[:, 1], c='r')
    plt.scatter(negative[:, 0], negative[:, 1], c='b')
    plt.show()


mu, sigma = estimate_gaussian(X)
best_epsilon, best_F1 = selectThreshold(Xval, yval, mu, sigma)
prediction(X, mu, sigma, best_epsilon)

# 这里出现一个经典的问题是浮点数没有办法比大小
# 另一个是概率小是阳性
# numpy数组间的逻辑运算用np.logical_and