#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:36:39 2018

@author: amar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

train_size = 0.8

X, y = 