import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
'''
path = 'data/ex6data1.mat'
data = loadmat(path)
X = data['X']
y = data['y']

Xpositive = X[y.ravel() == 1, :]
Xnegative = X[y.ravel() == 0, :]
plt.scatter(Xpositive[:, 0], Xpositive[:, 1], c='r', s=50)
plt.scatter(Xnegative[:, 0], Xnegative[:, 1], c='b', s=50)
plt.show()

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(X, y.ravel())
svc.score(X, y.ravel())

Confidence = svc.decision_function(X)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1], s=50, c=Confidence, cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()
print(svc)
'''


################################################################################
'''
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1-x2)**2)/(2*sigma**2))


path = 'data/ex6data2.mat'
data = loadmat(path)
X = data['X']
y = data['y']

Xpositive = X[y.ravel() == 1, :]
Xnegative = X[y.ravel() == 0, :]
plt.scatter(Xpositive[:, 0], Xpositive[:, 1], c='r', s=50)
plt.scatter(Xnegative[:, 0], Xnegative[:, 1], c='b', s=50)
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(X, y.ravel())
svc.score(X, y.ravel())
probability = svc.predict_proba(X)
# predict_proba会返回一个样本个数为行数，两列的array
# 第一列是样本为负（阴性）的概率，第二列是样本为正（阳性）的概率，两者之和为1
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1], s=30, c=probability[:, 1], cmap='Reds')
plt.show()
print(probability[300:400, :])
print(probability[300:400, 0]+probability[300:400, 1])
'''

##################################################################################
'''
path = 'data/ex6data3.mat'
data = loadmat(path)
X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval'].ravel()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        # 在训练集上训练
        score = svc.score(Xval, yval)
        # 在验证集上得到有效率

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)

# svm使用是给出代价函数上的C和核函数参数gamma然后去自动拟合所给的数据，得出theta
# theta参数被封装在svc内，要得到某个数据的预测结果，只要将数据输入到对象之中即可（调包侠真正奥义）
'''

##############################################################################################
path = 'data/spamTrain.mat'
data = loadmat(path)
X = data['X']  # (4000, 1899)
y = data['y'].ravel()  # (4000, 1)
path = 'data/spamTest.mat'
data = loadmat(path)
Xtest = data['Xtest']  # (1000, 1899)
ytest = data['ytest'].ravel()  # (1000, 1)

svc = svm.SVC()
svc.fit(X, y)
print(svc.score(Xtest, ytest))
