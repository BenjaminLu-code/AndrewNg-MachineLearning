import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

path = 'data/ex8data2.mat'
data = loadmat(path)
X = data['X']  # (1000, 11)
Xval = data['Xval']  # (100, 2)
yval = data['yval']  # (100, 1)


def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma


def selectThreshold(Xval, yval, mu, sigma):
    final_pro = np.ones(Xval.shape[0])
    for i in np.arange(Xval.shape[1]):
        mo = stats.norm(mu[i], sigma[i])
        pro = mo.pdf(Xval[:, i])
        final_pro = final_pro * pro
    best_epsilon = 0
    best_F1 = 0
    step = (final_pro.max() - final_pro.min()) / 100000
    for epsilon in np.arange(final_pro.min(), final_pro.max(), step):
        prediction = np.zeros(yval.shape)
        prediction[final_pro < epsilon] = 1
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
    final_pro = np.ones(X.shape[0])
    for i in np.arange(X.shape[1]):
        mo = stats.norm(mu[i], sigma[i])
        pro = mo.pdf(X[:, i])
        final_pro = final_pro * pro
    positive = X[final_pro < epsilon, :]
    print(positive.shape[0])


mu, sigma = estimate_gaussian(X)
best_epsilon, best_F1 = selectThreshold(Xval, yval, mu, sigma)
print(best_epsilon)
prediction(X, mu, sigma, best_epsilon)

# 这里出现一个经典的问题是浮点数没有办法比大小
# 另一个是概率小是阳性
# numpy数组间的逻辑运算用np.logical_and