# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 06:31:56 2020

@author: RaizQuadrada
"""

# This script is about a linear model (machine learning) built to make predictions.
# In this case, it will learn the population during some years of a city in
# Luxembourg called Habscht. After learning it, the model is going to predict
# the number of people living there for the next years.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Getting the dataset
dataset = pd.read_csv('CITYPOP-CITY_HABSCHTCAPLUXEMBOURG.csv')

# Splitting the dependent and independent variables
X_temp = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Collecting the years and tranforming it into integer numbers
X = [int(i[:4]) for i in X_temp]    

# Reshaping the matrix of features and dependent variable vector
X = np.reshape(X, -1, 1)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Setting up the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0, shuffle = False)

# Fitting a simple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making predictions for the test set
y_pred = regressor.predict(X_test)

# Comparing the model with the training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.axis(xmin=1980, ymin=1500)
plt.title('Population of Habscht (training set)')
plt.xlabel('Year')
plt.ylabel('Number of people')
plt.show()

# Comparing the model with the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.axis(xmin=1980, ymin=1500)
plt.title('Population of Habscht (test set)')
plt.xlabel('Year')
plt.ylabel('Number of people')
plt.show()

# Showing the real population growth during the time
plt.plot(X, y, color = 'green')
plt.axis(xmin=1980, ymin=1500)
plt.title('Population of Habscht')
plt.xlabel('Year')
plt.ylabel('Number of people')
plt.show()

# Conclusion: a linear model is a good fit for population growth time series
# analyses only when we are talking about a short period of time.
# Its graph, in fact, doesn't seen to have a linear relation between
# number of people and year of analysis, but a exponential relation - or
# another king of function.
# Then you must be very careful when using a linear model in this scenario.
