# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:58:32 2018

@author: Amar Enkhbat
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Importing data
# =============================================================================

X = np.array([[1, 1], [2, 1], [2, 2]], dtype = float)
y = np.array([0, 1, 2])

n = X.shape[1]
m = X.shape[0]
K = np.unique(y)

# =============================================================================
# Data Standardization
# =============================================================================
def mean(X):
    sum = 0.0
    for i in X:
        sum += i
    return sum / len(X)

def standard_deviation(X):
    sum = 0.0
    for i in X:
        sum += (i - mean(X)) ** 2.0
    return (sum / len(X)) ** 0.5

X_std = np.copy(X)     
X_std[:, 0] = (X[:, 0] - mean(X[:, 0])) / standard_deviation(X[:, 0])
X_std[:, 1] = (X[:, 1] - mean(X[:, 1])) / standard_deviation(X[:, 1])
#ones = np.ones(m)
X_std = np.column_stack((np.ones(m), X_std))
# =============================================================================
# Weight Initialization
# =============================================================================

rgen = np.random.RandomState(1)
w = rgen.normal(scale = 0.1, size = (n + 1, len(K)))

# =============================================================================
# Initial plot of data and decision boundaries
# =============================================================================

plt.scatter(X_std[y == 0, 1], X_std[y == 0, 2], label = "Class 0", marker = "o")
plt.scatter(X_std[y == 1, 1], X_std[y == 1, 2], label = "Class 0", marker = "x")
plt.scatter(X_std[y == 2, 1], X_std[y == 2, 2], label = "Class 0", marker = "*")
plt.show()

# =============================================================================
# Softmax function
# =============================================================================

#def softmax(z):

# =============================================================================
# Net sum function
# =============================================================================
def net_sum(X, w):
    return w.T.dot(X)