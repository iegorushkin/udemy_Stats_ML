# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:53:45 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# %%
# The foundation of this exercise
rng = np.random.default_rng()  # random generator
N = 50  # sample size
mu = 0.5  # true population mean
data = rng.normal(loc=mu, scale=1, size=N)
# data = np.random.randn(N) + mu

# plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Visualization of our sample', y=0.93)
ax.plot(data, 'ko', markerfacecolor='w', markersize=10)
ax.set(xlabel='Data index', ylabel='Data value')
# %%
# Let's work with some range of standard deviations

# parameters
N = 50  # sample size
mu = 0.5  # true population mean
h0_val = 0  # the null hypothesis value
stds = np.linspace(0.1, 3, 500)
# arrays for storing t-statistics and p-values
t_array = np.zeros(len(stds))
p_array = np.zeros(len(stds))

# static dataset
# data = rng.standard_normal(size=N)
# data = np.random.randn(N)

# loop through every std
for i in range(len(stds)):
    data = rng.normal(loc=mu, scale=stds[i], size=N)
    # data = stds[i]*np.random.randn(N) + mu
    # data = stds[i]*data + mu
    t_array[i], p_array[i] = stats.ttest_1samp(data, h0_val) # 2-tail t-test
# %%
# Visualization!

# create figure and define gridspec
fig = plt.figure(figsize=(14, 8), layout='constrained')
grid = fig.add_gridspec(2, 2)
# add subplots
ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, :])

# ax1
ax1.plot(stds, t_array, 'ko', markerfacecolor='w', markersize=5)
ax1.set(title='T-values versus standard deviation',
        xlabel='Standard deviation', ylabel='T-value')

# ax2
ax2.plot(stds, p_array, 'k*', markerfacecolor='b', markersize=5)
ax2.set(title='P-values versus standard deviation',
        xlabel='Standard deviation', ylabel='P-value')

# ax3
ax3.plot(t_array, p_array, 'k^', markerfacecolor='m', markersize=5)
ax3.set(title='T-values versus P-values', xlabel='T-value', ylabel='P-value')
