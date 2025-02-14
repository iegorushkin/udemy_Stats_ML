# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:28:50 2021

@author: Igor
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
"""
Compare mean and median for distributions with:
1) different sample sizes
2) different amplitude of outliers
"""
#%%

# Parameters
mu = 3
std = 0.5
# Number of samples
#n = 2001
n = 101
# Number of outliers
m = 100
bins = 40
# Random Generator
rng = np.random.default_rng()

# Data
normal_data = std*rng.standard_normal(n) + mu
lognormal_data = np.exp(normal_data)

# Median and mean
mean1 = np.mean(normal_data)
mean2 = np.mean(lognormal_data)
median1 = np.median(normal_data)
median2 = np.median(lognormal_data)

# Define outliers
outliers1 = np.zeros(m)
outliers2 = np.zeros(m)
outliers1 = np.max(normal_data) + 10*rng.uniform(size=m)  # weaker
outliers2 = np.max(lognormal_data) + 100*rng.uniform(size=m)  # stronger
#%%
# Create histogram
hist1, bin_edges1 = np.histogram(normal_data, bins)
x1 = (bin_edges1[1:] + bin_edges1[:-1])/2
hist2, bin_edges2 = np.histogram(lognormal_data, bins)
x2 = (bin_edges2[1:] + bin_edges2[:-1])/2

#plt.style.use('seaborn-whitegrid')
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 6))
fig1.suptitle('Vanilla normal and lognormal distributions', y=0.96)
# Plot vanilla distributions
ax1[0].plot(x1, hist1, 'r', lw=2,)
ax1[0].plot([mean1, mean1], [0, np.max(hist1)], 'b--', lw=2,
            label=f'mean = {np.round(mean1, 2)}')
ax1[0].plot([median1, median1], [0, np.max(hist1)], 'g--', lw=2,
            label=f'median = {np.round(median1, 2)}')
ax1[0].set(title='normal distribution', xlabel='values', ylabel='counts')
ax1[0].legend(loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax1[0].grid()

ax1[1].plot(x2, hist2, 'c', lw=2,)
ax1[1].plot([mean2, mean2], [0, np.max(hist2)], 'b--', lw=2,
            label=f'mean = {np.round(mean2, 2)}')
ax1[1].plot([median2, median2], [0, np.max(hist2)], 'g--', lw=2,
            label=f'median = {np.round(median2, 2)}')
ax1[1].set(title='lognormal distribution', xlabel='values', ylabel='counts')
ax1[1].legend(loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax1[1].grid()
#%%
'''small outliers'''

# Add outliers to data
snormal_data = np.hstack((normal_data, outliers1))
slognormal_data = np.hstack((lognormal_data, outliers1))

# Median and mean
mean1 = np.mean(snormal_data)
mean2 = np.mean(slognormal_data)
median1 = np.median(snormal_data)
median2 = np.median(slognormal_data)

# Create histogram
hist1, bin_edges1 = np.histogram(snormal_data, bins)
x1 = (bin_edges1[1:] + bin_edges1[:-1])/2
hist2, bin_edges2 = np.histogram(slognormal_data, bins)
x2 = (bin_edges2[1:] + bin_edges2[:-1])/2

fig2, ax2 = plt.subplots(1, 2, figsize=(10, 6))
fig2.suptitle('Normal and lognormal distributions with smaller outliers',
              y=0.96)
# Plot distributions
ax2[0].plot(x1, hist1, 'r', lw=2,)
ax2[0].plot([mean1, mean1], [0, np.max(hist1)], 'b--', lw=2,
            label=f'mean = {np.round(mean1, 2)}')
ax2[0].plot([median1, median1], [0, np.max(hist1)], 'g--', lw=2,
            label=f'median = {np.round(median1, 2)}')
ax2[0].set(title='normal distribution', xlabel='values', ylabel='counts')
ax2[0].legend(loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax2[0].grid()

ax2[1].plot(x2, hist2, 'c', lw=2,)
ax2[1].plot([mean2, mean2], [0, np.max(hist2)], 'b--', lw=2,
            label=f'mean = {np.round(mean2, 2)}')
ax2[1].plot([median2, median2], [0, np.max(hist2)], 'g--', lw=2,
            label=f'median = {np.round(median2, 2)}')
ax2[1].set(title='lognormal distribution', xlabel='values', ylabel='counts')
ax2[1].legend(loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax2[1].grid()
#%%
'''large outliers'''

# Add outliers to data
snormal_data = np.hstack((normal_data, outliers2))
slognormal_data = np.hstack((lognormal_data, outliers2))

# Median and mean
mean1 = np.mean(snormal_data)
mean2 = np.mean(slognormal_data)
median1 = np.median(snormal_data)
median2 = np.median(slognormal_data)

# Create histogram
hist1, bin_edges1 = np.histogram(snormal_data, bins)
x1 = (bin_edges1[1:] + bin_edges1[:-1])/2
hist2, bin_edges2 = np.histogram(slognormal_data, bins)
x2 = (bin_edges2[1:] + bin_edges2[:-1])/2

fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6))
fig3.suptitle('Normal and lognormal distributions with larger outliers',
              y=0.96)
# Plot distributions
ax3[0].plot(x1, hist1, 'r', lw=2,)
ax3[0].plot([mean1, mean1], [0, np.max(hist1)], 'b--', lw=2,
            label=f'mean = {np.round(mean1, 2)}')
ax3[0].plot([median1, median1], [0, np.max(hist1)], 'g--', lw=2,
            label=f'median = {np.round(median1, 2)}')
ax3[0].set(title='normal distribution', xlabel='values', ylabel='counts')
ax3[0].legend( loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax3[0].grid()

ax3[1].plot(x2, hist2, 'c', lw=2,)
ax3[1].plot([mean2, mean2], [0, np.max(hist2)], 'b--', lw=2,
            label=f'mean = {np.round(mean2, 2)}')
ax3[1].plot([median2, median2], [0, np.max(hist2)], 'g--', lw=2,
            label=f'median = {np.round(median2, 2)}')
ax3[1].set(title='lognormal distribution', xlabel='values', ylabel='counts')
ax3[1].legend(loc='upper right', frameon=True, framealpha=1, fontsize=12)
ax3[1].grid()
'''
The median is much less sensitive to the presence of outliers,
much less affected by them than the mean.

The mean is very sensitive to outliers,
especially if the outlier is large enough
'''
