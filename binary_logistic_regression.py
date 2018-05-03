#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:36:39 2018

@author: amar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# =============================================================================
# Importing data
# =============================================================================

X = np.array([[1, 1], [2, 1], [1, 5], [6, 4], [5, 5], [6, 5]], dtype = float)
y = np.array([0, 0, 1, 1, 1, 1])

iris = datasets.load_iris()
X = iris.data[:, 0:2]
y = iris.target

X = X[y != 2]
y = y[y != 2]


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

# =============================================================================
# Weight Initialization
# =============================================================================

rgen = np.random.RandomState(1)
weight = rgen.normal(scale = 1, size = X.shape[1] + 1)

# =============================================================================
# Activation function
# =============================================================================

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# =============================================================================
# Activate
# =============================================================================
def activate(x):
    return sigmoid(x)
# =============================================================================
# Plot initial decision boundary
# =============================================================================

plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], label = "Class 0")
plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], label = "Class 1")
t = np.arange(-2, 2, 0.1)
plt.plot(t, (-weight[0] - weight[1] * t)/weight[2], label = "Decision boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# =============================================================================
# Classification
# =============================================================================

# Learning Rate
eta = 0.01

# Number of iterations
epoch = 10000

# Regularization coef
cost_lambda = 0.1

# Cost array
cost_series = []

# Fitting/Training
for i in range(epoch):
    # Dot product of z (= w1*x1 + w2*x2 + ... + w6*x6) + w0
    net_input = np.dot(X_std, weight[1:]) + weight[0]
    # Cost function (Cross Entropy Function)
    cost = ((y.dot((np.log(activate(net_input)))) + (1 - y).dot(np.log(1 - activate(net_input)))) + cost_lambda * (weight[1:]**2).sum()).sum() / (-X.shape[1])
    cost_series.append(cost)
    
    # Weight update
    weight[1:] += eta * np.dot((y - activate(net_input)), X_std)
    weight[0] += eta * np.sum(y - activate(net_input))
    
    if(i > 2):
        if (cost_series[-2] - cost_series[-1]) < 0.001:
            break
    # Plot decision boundary
#    plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], label = "Class 0")
#    plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], label = "Class 1")
#    t = np.arange(-2, 2, 0.1)
#    plt.plot(t, (-weight[0] - weight[1] * t)/weight[2])
#    plt.legend()
#    plt.show()
    
cost = np.array(cost)
plt.plot(range(len(cost_series)), cost_series, label = "Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], label = "Class 0")
plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], label = "Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
t = np.arange(-2, 2, 0.1)
plt.plot(t, (-weight[0] - weight[1] * t)/weight[2], label = "Decision boundary")
plt.legend()
plt.show()
print("Suggested epoch number:", len(cost_series))