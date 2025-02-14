# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:26:04 2021

@author: Igor
"""
# imports
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# simulation parameters
rng = np.random.default_rng()
N = 500  # Number of samples
data = np.zeros((N, 4))  # rows - observations, columns - channels
# imposed empirical correlation coefficients between 1 channel and other channels
r = np.array([0.3, 0.5, 0.75,])
# %%
# simulate the data
data[:, 0] = rng.normal(loc=0, scale=1.2, size=N)
for i in range(1, 4):
    # Mike's formula implementation     
    data[:, i] = (data[:, 0]*r[i-1]
                  + (rng.normal(loc=0, scale=2, size=N)*np.sqrt(1 - r[i-1]**2)))
# %%
# calculate the correlation and covariance matrices
corr_matrix = np.zeros((4, 4))
cov_matrix = np.zeros((4, 4))

corr_matrix = np.corrcoef(data, rowvar=False)
cov_matrix = np.cov(data, ddof=1, rowvar=False)
# %%
# covariance matrix from correlation matrix via linear algebra formula

# compute the diagnal standard deviation matrix
stds = [np.std(data[:, i], ddof=1) for i in range(4)]
stds_matrix = np.diag(stds)

# now, covariance matrix:
# Can use @ instead of np.matmul
# cov_matrix_f = stds_matrix @ corr_matrix @ stds_matrix
cov_matrix_f = np.matmul(stds_matrix, corr_matrix)
cov_matrix_f = np.matmul(cov_matrix_f, stds_matrix)

# check the difference between 2 covariance matrices
difference_1 = np.abs(cov_matrix - cov_matrix_f)
print(difference_1)  # almost zero

#%% 
# Correlation matrix from covariance matrix
corr_matrix_f = np.linalg.inv(stds_matrix) @ cov_matrix @ np.linalg.inv(stds_matrix)
difference_2 = np.abs(corr_matrix_f - corr_matrix)
print(difference_2)  # almost zero
