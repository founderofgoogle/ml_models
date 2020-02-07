# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:07:43 2020

@author: RaizQuadrada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importing the datasets
dataset_angola = pd.read_csv('WPOV-AGO_SI_POV_NOP5.csv')
dataset_brazil = pd.read_csv('WPOV-BRA_SI_POV_NOP5.csv')
dataset_china = pd.read_csv('WPOV-CHN_SI_POV_NOP5.csv')
dataset_mexico = pd.read_csv('WPOV-MEX_SI_POV_NOP5.csv')
dataset_usa = pd.read_csv('WPOV-USA_SI_POV_NOP5.csv')

# Separating the dependent and independent variables
# Angola
X_a = dataset_angola.iloc[:, 0].values
y_ang = dataset_angola.iloc[:, 1].values

# Brazil
X_b = dataset_brazil.iloc[:, 0].values
y_bra = dataset_brazil.iloc[:, 1].values

# China
X_c = dataset_china.iloc[:, 0].values
y_chi = dataset_china.iloc[:, 1].values

# Mexico
X_m = dataset_mexico.iloc[:, 0].values
y_mex = dataset_mexico.iloc[:, 1].values

# United States of America
X_u = dataset_usa.iloc[:, 0].values
y_usa = dataset_usa.iloc[:, 1].values

# Cleaning the date (collecting only the year)
X_ang = [int(i[:4]) for i in X_a]
X_bra = [int(i[:4]) for i in X_b]
X_chi = [int(i[:4]) for i in X_c]
X_mex = [int(i[:4]) for i in X_m]
X_usa = [int(i[:4]) for i in X_u]

# Coverting the year list into an array
X_ang = np.array(X_ang)
X_bra = np.array(X_bra)
X_chi = np.array(X_chi)
X_mex = np.array(X_mex)
X_usa = np.array(X_usa)

# Reshaping the data (dependent variable)
X_ang = X_ang.reshape(-1, 1)
X_bra = X_bra.reshape(-1, 1)
X_chi = X_chi.reshape(-1, 1)
X_mex = X_mex.reshape(-1, 1)
X_usa = X_usa.reshape(-1, 1)

# Reshaping the data (independent variable)
y_ang = y_ang.reshape(-1, 1)
y_bra = y_bra.reshape(-1, 1)
y_chi = y_chi.reshape(-1, 1)
y_mex = y_mex.reshape(-1, 1)
y_usa = y_usa.reshape(-1, 1)

# Changing the order of the data in the matrix of features (from oldest to newest)
X_ang = np.flip(X_ang)
X_bra = np.flip(X_bra)
X_chi = np.flip(X_chi)
X_mex = np.flip(X_mex)
X_usa = np.flip(X_usa)

# Changing the order of the data in the dependent vector variable (from oldest to newest)
y_ang = np.flip(y_ang)
y_bra = np.flip(y_bra)
y_chi = np.flip(y_chi)
y_mex = np.flip(y_mex)
y_usa = np.flip(y_usa)

# Creating the simple linear models
ang_regressor = LinearRegression()
bra_regressor = LinearRegression()
chi_regressor = LinearRegression()
mex_regressor = LinearRegression()
usa_regressor = LinearRegression()

# Training the models
ang_regressor.fit(X_ang, y_ang)
bra_regressor.fit(X_bra, y_bra)
chi_regressor.fit(X_chi, y_chi)
mex_regressor.fit(X_mex, y_mex)
usa_regressor.fit(X_usa, y_usa)

# Comparing the model with the real data
plt.scatter(X_ang, y_ang, color = 'blue')
plt.plot(X_ang, y_ang, color = 'purple', label = 'Real data')
plt.plot(X_ang, ang_regressor.predict(X_ang), color = 'black', label = 'Linear model')
plt.title('US$5 earning in Angola')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

plt.scatter(X_bra, y_bra, color = 'blue')
plt.plot(X_bra, y_bra, color = 'purple', label = 'Real data')
plt.plot(X_bra, bra_regressor.predict(X_bra), color = 'black', label = 'Linear model')
plt.title('US$5 earning in Brazil')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

plt.scatter(X_chi, y_chi, color = 'blue')
plt.plot(X_chi, y_chi, color = 'purple', label = 'Real data')
plt.plot(X_chi, chi_regressor.predict(X_chi), color = 'black', label = 'Linear model')
plt.title('US$5 earning in China')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

plt.scatter(X_mex, y_mex, color = 'blue')
plt.plot(X_mex, y_mex, color = 'purple', label = 'Real data')
plt.plot(X_mex, mex_regressor.predict(X_mex), color = 'black', label = 'Linear model')
plt.title('US$5 earning in Mexico')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

plt.scatter(X_usa, y_usa, color = 'blue')
plt.plot(X_usa, y_usa, color = 'purple', label = 'Real data')
plt.plot(X_usa, usa_regressor.predict(X_usa), color = 'black', label = 'Linear model')
plt.title('US$5 earning in USA')
plt.xlabel('Year')
plt.ylabel('Millions of people')
plt.legend()
plt.show()

# Conclusion: if a linear model doesn't fit the data pretty well
# it can be very useful to show the tendency of it because
# this model is simple and, therefore, requires relatively
# few computational power to be trained.
