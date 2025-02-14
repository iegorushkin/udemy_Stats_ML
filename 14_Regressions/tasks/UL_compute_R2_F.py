# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:50:47 2021

@author: Igor
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Investigate the effects of sleep on food spending
# При помощи часов сна (независимая переменная)
# объясняем траты на еду (зависимая переменная)

# Generate the data and plot them
sleep_hours = np.array([5, 5.5, 6, 6, 7, 7, 7.5, 8, 8.5, 9])
dollars = np.array([47, 53, 52, 44, 39, 49, 50, 38, 43, 40])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sleep_hours, dollars, '*k', markersize=12)
ax.set(title='Visualization of the raw data', xlabel='Sleep hours',
       ylabel='Dollars')
ax.grid()
# %%
# Simple regression ("manual")

# 1) Create the design matrix
des_matrix = np.array((np.ones(len(dollars)), sleep_hours)).T
# 2) Calculate left inverse of des_matrix
# @ is the same operator as numpy.matmul
li = (np.linalg.inv(des_matrix.T@des_matrix))@des_matrix.T
# 3) Get regression coefficients
betas = li@dollars
# 4) Use coefficients to obtain model values
dollars_model = betas[0] + sleep_hours*betas[1]
# 5) Compare modeled and observed values
fig, ax = plt.subplots(figsize=(10, 6))
# observed data
ax.plot(sleep_hours, dollars, '*k', markersize=12)
# predicted results
ax.plot(sleep_hours, dollars_model, 'o-r', markersize=12)
# show the errors (residuals)
for i in range(len(dollars)):
    ax.plot([sleep_hours[i], sleep_hours[i]], [dollars[i], dollars_model[i]],
            'm--')
ax.set(title='Simple regression', xlabel='Sleep hours', ylabel='Dollars')
ax.legend(('Data', 'Model', 'Residuals'))
ax.grid()
# %%
# Evaluating a model fit with R squared
R2 = 1 - (np.sum((dollars - dollars_model)**2)
          / np.sum((dollars - np.mean(dollars))**2))
print(f"R squared: {R2}")

'''
NOTE:
In linear least squares multiple regression with an estimated intercept term,
R2 equals the square of the Pearson correlation coefficient between the observed
y and modeled (predicted) f data values of the dependent variable.
'''
# %%
# Evaluating a model statistical significance with F-test

# degrees of freedom
model_df = len(betas) - 1  # number of features minus one
res_df = len(dollars) - len(betas)  # number of observations minus number of features
# sums of squares
SS_model = np.sum((dollars_model - np.mean(dollars))**2)
SS_res = np.sum((dollars - dollars_model)**2)
# mean squares
MS_model = SS_model / model_df
MS_res = SS_res / res_df
# F-test
F = MS_model / MS_res
# Значение распределения Фишера (при заданных степенях свободы),
# правее которого  лежит 5% данных
p_threshold = stats.f.ppf(0.95, model_df, res_df)
# Пи-значение (неформально): какова вероятность того, что произойдёт событие,
# F-значение которого больше полученного в наших рассчётах.
p_value = 1 - stats.f.cdf(F, model_df, res_df)
print(f'P-value: {p_value}\nP-threshold: 0.05')
