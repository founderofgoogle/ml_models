# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 06:23:15 2020

@author: RaizQuadrada
"""

# The script below builds a linear model to learn about how much of the population
# (in percentage) from age 15 to 24 works in South Sudan.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('WADI-SSD_SL_TLF_ACTI_1524_ZS.csv')

# Separating the matrix of features and the dependent vector variable
# The matrix of vectors needs to be cleaned and coverted to integer.
X_temp = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Cleaning and converting the independent variable
X = []
for i in X_temp:
    X.append(int(i[:4]))

# Reshaping the data
X = np.reshape(X, -1, 1)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Reversing the order of elements. It will make our model to use
# the past data to predict the new one. Here past data means
# that we are talking about the initial years - since this model
# works with time series data.
X = np.flip(X)
y = np.flip(y)

# Creating the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0, shuffle = False)

# Creating a simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making predictions for the test set
y_pred = regressor.predict(X_test)

# Comparing the model with the training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('Years')
plt.ylabel('% of the population ages 15-24 of South Sudan')
plt.title('Labor force participation rate (training set)')
plt.show()

# Comparting the model with the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.xlabel('Years')
plt.ylabel('% of the population ages 15-24 of South Sudan')
plt.title('Labor force participation rate (test set)')
plt.show()

# The real data plotted with the linear model
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Years')
plt.ylabel('% of the population ages 15-24 of South Sudan')
plt.title('Labor force participation rate (all the data)')
plt.show()

# Conclusion: the data doesn't fit a linear model. So it was a wrong modeling and can't
# make predictions with accuracy. That's why other kinds of machile learning models
# must be applied.