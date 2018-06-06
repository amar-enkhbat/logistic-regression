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
eta = 0.1

# Max iterations
epoch = 10000

# Regularization coeffecient
cost_lambda = 10

# Cost vector
cost_series = []

# Fitting/Training
for i in range(epoch):
    # Dot product of z (= w1*x1 + w2*x2 + ... + w6*x6) + w0
    net_input = np.dot(X_std, weight[1:]) + weight[0]
    # Cost function (Cross Entropy Function)
    regularization = (cost_lambda * (weight[1:]**2).sum()) / (2 * X.shape[1])
    cost = -y.dot(np.log(activate(net_input))) - ((1 - y).dot(np.log(1 - activate(net_input)))) + regularization
    cost_series.append(cost)
    
    # Weight update with regularization
    weight[0] -= eta * np.sum(activate(net_input) - y)
    weight[1:] -= eta * (np.dot((activate(net_input) - y), X_std) + cost_lambda * weight[1:] / X.shape[1])
        
    # Convergence condition
    if(i > 2):
        if (cost_series[-2] - cost_series[-1]) < 0.001:
            break
        
# =============================================================================
# Cost function plot
# =============================================================================

plt.plot(range(len(cost_series)), cost_series, label = "Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

print("Suggested epoch number:", len(cost_series))

# =============================================================================
# Final decision boundary plot
# =============================================================================

plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], label = "Class 0")
plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], label = "Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
t = np.arange(-2, 2, 0.1)
plt.plot(t, (-weight[0] - weight[1] * t)/weight[2], label = "Decision boundary")
plt.legend()
plt.show()
