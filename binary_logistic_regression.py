#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:36:39 2018

@author: amar
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Importing data
# =============================================================================

X = np.array([[1, 1], [2, 1], [1, 5], [6, 4], [5, 5], [6, 5]], dtype = float)
y = np.array([0, 0, 1, 1, 1, 1])


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

X_std = np.copy(X.astype(float))     
X_std[:, 0] = (X[:, 0] - mean(X[:, 0])) / standard_deviation(X[:, 0])
X_std[:, 1] = (X[:, 1] - mean(X[:, 1])) / standard_deviation(X[:, 1])


# =============================================================================
# Graph Data
# =============================================================================


# =============================================================================
# Separating data
# =============================================================================
rgen = np.random.RandomState(1)
weight = rgen.normal(scale = 1, size = X.shape[1] + 1)

# =============================================================================
# Activation function
# =============================================================================

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-np.clip(x, -250, 250)))

# =============================================================================
# Activate
# =============================================================================
def activate(x):
    return sigmoid(x)

eta = 0.1
epoch = 100

plt.scatter(X_std[:3, 0], X_std[:3, 1], label = "Class 0")
plt.scatter(X_std[3:, 0], X_std[3:, 1], label = "Class 1")
t = np.arange(-2, 2, 0.1)
plt.plot(t, (-weight[0] - weight[1] * t)/weight[2])
plt.legend()
plt.show()

for i in range(epoch):
    net_input = np.dot(X_std, weight[1:])
    cost = -(y * (np.log(activate(net_input))) + (1 - y) * np.log(1 - activate(net_input)))
#    print("Error = ", np.average(y - activate(net_input)))
    weight[1:] += eta * np.dot((y - activate(net_input)), X_std)
    weight[0] += eta * np.sum(y - activate(net_input))
    plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], label = "Class 0")
    plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], label = "Class 1")
#    plt.xlim(-2, 2)
#    plt.ylim(-2, 2)
    t = np.arange(-2, 2, 0.1)
    plt.plot(t, (-weight[0] - weight[1] * t)/weight[2])
    plt.legend()
    plt.show()
    print(weight)

    
