# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:26:04 2021

@author: Igor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# simulation parameters
N = 500  # Number of samples
data = np.zeros((4, N))  # rows - channels, columns - observations
# imposed empirical correlation coefficients between 1 channel and other channels
r = np.array([0.4, 0.7, 0.8,])
# %%
# simulate the data
data[0, :] = np.random.default_rng().normal(loc=0, scale=2, size=N)
for i in range(1, 4):
    data[i, :] = (data[0, :]*r[i-1]
                  + (np.random.default_rng().normal(loc=0, scale=2, size=N)
                     *np.sqrt(1 - r[i-1]**2)))
# %%
# calculate the correlation and covariance matrices
corr_matrix = np.zeros((4, 4))
cov_matrix = np.zeros((4, 4))

corr_matrix = np.corrcoef(data)
cov_matrix = np.cov(data)
# %%
# covariance matrix from correlation matrix via linear algebra formula

# compute the diagnal standard deviation matrix
stds = [np.std(data[i, :], ddof=1) for i in range(4)]
stds_matrix = np.diag(stds)

# now, covariance matrix:
# Can use @ instead of np.matmul
# cov_matrix_f = stds_matrix @ corr_matrix @ stds_matrix
cov_matrix_f = np.matmul(stds_matrix, corr_matrix)
cov_matrix_f = np.matmul(cov_matrix_f, stds_matrix)

# check the difference between 2 covariance matrices
difference = np.abs(cov_matrix - cov_matrix_f)
print(difference)
