# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:34:30 2020

@author: RaizQuadrada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('WPOV-URY_SI_POV_GAP5.csv')

# Separating the matrix of features and the dependent vector variable
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Clearning the matrix of features
X_clean = [int(i[:4]) for i in X]

# Changing the order of the data
X_clean = np.flip(X_clean)
y = np.flip(y)

# Transforming the matrix of features into an array
X_clean = np.array(X_clean)

# Reshaping the data
X_clean = X_clean.reshape(-1, 1)
y = y.reshape(-1, 1)

# Training a polynomial model
pol_regressor = PolynomialFeatures(degree = 3)
X_poly = pol_regressor.fit_transform(X_clean)
lin_regressor = LinearRegression()
lin_regressor.fit(X_poly, y)

# Making predictions
y_pred = lin_regressor.predict(pol_regressor.fit_transform(X_clean))

# Making a more accurate predictions
X_grid = np.arange(min(X_clean), max(X_clean), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
y_pred_2 = lin_regressor.predict(pol_regressor.fit_transform(X_grid))

# Showing the actual data
plt.plot(X_clean, y, color = 'purple', marker = 'o', label = 'Real data')
plt.plot(X_clean, y_pred, color = 'green', label = '3th degree polynomial')
plt.plot(X_grid, y_pred_2 , color = 'orange', label = 'Accurate polynomial model')
plt.title('US$5 earning in Paraguay')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

# Conclusion: in this dataset, the third degree polynomial is the most
# simple polynomial that show clearly the tendendy of the data along
# the time. In this case, I didn't built anything to make predictions.
