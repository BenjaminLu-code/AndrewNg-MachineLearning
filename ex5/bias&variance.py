import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('ex5data1')
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']
print(X.shape, y.shape, Xtest.shape, ytest.shape, Xval.shape, yval.shape)
plt.scatter(X, y)
plt.show()