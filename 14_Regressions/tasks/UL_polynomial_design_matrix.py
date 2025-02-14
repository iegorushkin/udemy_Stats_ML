# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 23:18:53 2021

@author: Igor
"""
import numpy as np
import matplotlib.pyplot as plt

# generate the data

n = 100
x = np.linspace(-4, 4, n)
y1 = x**2 + np.random.default_rng().normal(loc=0, scale=1.5, size=n)
y2 = x**3 + np.random.default_rng().normal(loc=0, scale=1.75, size=n)

# plot the data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, 'ko', markerfacecolor='r')
ax.plot(x, y2, 'ko', markerfacecolor='g')
ax.set(xlabel='x', ylabel='y',
       title='Examples of functions based on second and third order dependencies')
ax.legend(('Quadratic', 'Cubic'))
# %%
# Polynomial regression ("manual") for second and third order based data

# 1) Create the design matrix
des_matrix1 = np.array((np.ones(n), x, x**2)).T
des_matrix2 = np.array((np.ones(n), x, x**2, x**3)).T
# 2) Calculate left inverse of des_matrix
# @ is the same operator as numpy.matmul
li1 = (np.linalg.inv(des_matrix1.T@des_matrix1))@des_matrix1.T
li2 = (np.linalg.inv(des_matrix2.T@des_matrix2))@des_matrix2.T
# 3) Get regression coefficients
betas1 = li1@y1
betas2 = li2@y2
# 4) Use coefficients to obtain model values
y1_model = betas1[0] + betas1[1]*x + betas1[2]*x**2
y2_model = betas2[0] + betas2[1]*x + betas2[2]*x**2 + betas2[3]*x**3
# 5) Compare modeled and observed values
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("Polynomial regression (second and third orders)", y=0.95)
# y1
# observed data
ax[0].plot(x, y1, '*k', markersize=10)
# predicted results
ax[0].plot(x, y1_model, '-r')
ax[0].set(xlabel='x', ylabel='y')
ax[0].legend(('y1', 'y1 fit'))
ax[0].grid()
# y2
# observed data
ax[1].plot(x, y2, '*k', markersize=10)
# predicted results
ax[1].plot(x, y2_model, '-r')
ax[1].set(xlabel='x', ylabel='y')
ax[1].legend(('y2', 'y2 fit'))
ax[1].grid()
# %%
# Polynomial regression ("built-in") for second and third order based data
# for y1
pterms1 = np.polyfit(x, y1, 2)  # highest power terms first!
yHat1 = np.polyval(pterms1, x)

# for y2
pterms2 = np.polyfit(x, y2, 3)
yHat2 = np.polyval(pterms2, x)
# %%
# Compare the manual and built-in polynomial regression results
print(f"Regressors for y1.\nManually calculated: {betas1}" +
      f"\nObtained from numpy function: {pterms1[::-1]}")
print(f"\nRegressors for y2.\nManually calculated: {betas2}" +
      f"\nObtained from numpy function: {pterms2[::-1]}")
