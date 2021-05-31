import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

path = 'data/me.mat'
data = loadmat(path)
I = data['I']
print(I.shape)