# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 07:24:47 2020

@author: RaizQuadrada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('WPOV-PRY_SI_POV_NOP5.csv')

# Separating the dependent and the independent variables
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Collecting just the years
X_clean = [int(i[:4]) for i in X]

# Changing the order of the data
# The first ones are the oldest and the last ones the newest
X_clean = np.flip(X_clean)
y = np.flip(y)

# Converting the new list into an array
X_clean = np.array(X_clean)

# Reshaping the data
X_clean = X_clean.reshape(-1, 1)
y = y.reshape(-1, 1)

# Separating the training set and the test one
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, train_size = 0.8, random_state = 0, shuffle = False)

# Training a linear model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Finding the predictions of the polynomial model
y_pred_lin = regressor.predict(X_test)

# Training a polynomial model
pol_regressor = PolynomialFeatures(degree = 3)
X_poly = pol_regressor.fit_transform(X_train)
linear_regressor = LinearRegression()
linear_regressor.fit(X_poly, y_train)

# Finding the predictions of the polynomial model
y_pred = linear_regressor.predict(pol_regressor.fit_transform(X_test))

# Plotting the result
plt.scatter(X_clean, y, color = 'purple')
plt.plot(X_clean, y, color = 'purple', label = 'Real data')
plt.title('US$5 earning in Paraguay')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

plt.scatter(X_test, y_test, color = 'blue', label = 'Real test data')
plt.plot(X_test, y_pred, color = 'red', label = 'Polynomial model (3th degree)')
plt.plot(X_test, y_pred_lin, color = 'orange', label = 'Linear model')
plt.title('US$5 earning in Paraguay')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

# Conclusion: the linear model was able to catch the major tendency of the
# data in the training set. However, it is the opposite of the tendency
# of the test set. In this scenario, the third degree polynomial is much
# better. Growing the polynomial's degree didn't change the prediction
# meaningfully. Then, this polynomial is the best option once every
# complexity must be avoided (Occam's razor).
