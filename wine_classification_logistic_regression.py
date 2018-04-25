#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:05:02 2018

@author: amar
"""

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Importing data
# =============================================================================

columns = ["Class", "Alcohol", "Malic acid", 
           "Ash", "Alcalinity of ash", "Magnesium", 
           "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
           "Proanthocyanins", "Color intensity", "Hue", 
           "OD280/OD315 of diluted wines", "Proline"]
wine = pd.read_csv('wine_train.data')
wine.columns = columns

# =============================================================================
# Separating data
# =============================================================================

from sklearn.model_selection import train_test_split

X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 1)
plt.hist(X_train)
plt.title("Histogram of raw data")
plt.legend()
plt.show()

# =============================================================================
# Data Standardization
# =============================================================================

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

plt.hist(X_train_std)
plt.title("Histogram of standardised data")
plt.legend()
plt.show()

# =============================================================================
# SVM regression with Grid-Search without standardization
# =============================================================================
from sklearn import metrics, grid_search
from sklearn.linear_model import LogisticRegression

param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000]
        }

print("Classifying without standardization")

classifier = grid_search.GridSearchCV(LogisticRegression(), param_grid)
classifier.fit(X_train, y_train)

print("\nBest Estimator:\n%s\n"% classifier.best_estimator_)
#for params, mean_score, all_scores in classifier.grid_scores_:
#    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

expected = y_test
predicted = classifier.predict(X_test)

print("Accuracy:\n", metrics.accuracy_score(expected, predicted))
print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted))
print("Precision:\n", metrics.precision_score(expected, predicted, average = None))
print("Recall:\n", metrics.recall_score(expected, predicted, average = None))
print("F-measure:\n", metrics.f1_score(expected, predicted, average = None))

# =============================================================================
# SVM regression with Grid-Search with standardization
# =============================================================================

print("Classifying with standardization")

classifier = grid_search.GridSearchCV(LogisticRegression(), param_grid)
classifier.fit(X_train_std, y_train)

print("\nBest Estimator:\n%s\n"% classifier.best_estimator_)
#for params, mean_score, all_scores in classifier.grid_scores_:
#    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

expected = y_test
predicted = classifier.predict(X_test_std)

print("Accuracy:\n", metrics.accuracy_score(expected, predicted))
print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted))
print("Precision:\n", metrics.precision_score(expected, predicted, average = None))
print("Recall:\n", metrics.recall_score(expected, predicted, average = None))
print("F-measure:\n", metrics.f1_score(expected, predicted, average = None))

# =============================================================================
# Testing with new test data
# =============================================================================
wine_test = pd.read_csv('wine_test.data')
X_test, y_test = wine_test.iloc[:, 1:].values, wine_test.iloc[:, 0].values
X_test_std = stdsc.transform(X_test)

print("\nBest Estimator:\n%s\n"% classifier.best_estimator_)
#for params, mean_score, all_scores in classifier.grid_scores_:
#    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

expected = y_test
predicted = classifier.predict(X_test_std)

print("Accuracy:\n", metrics.accuracy_score(expected, predicted))
print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted))
print("Precision:\n", metrics.precision_score(expected, predicted, average = None))
print("Recall:\n", metrics.recall_score(expected, predicted, average = None))
print("F-measure:\n", metrics.f1_score(expected, predicted, average = None))

# =============================================================================
# Best parameters
# =============================================================================
print("Best value for hyperparameter C is 0.1")