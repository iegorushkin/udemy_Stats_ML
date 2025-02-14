# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:18:49 2021

@author: Igor
"""
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def scale_to_interval(data, interval):
    '''
    Changes the data range to that specified in the parameter interval
    '''
    new_data = (interval[0] + (data - np.min(data))*(interval[1] - interval[0])
                / (np.max(data) - np.min(data)))
    return np.round(new_data, 0)


rng = np.random.default_rng()
n_cat = 10  # number of categories
n_values = 1000  # number of values in each sample
r_d_array = np.linspace(-1, 1, 51)  # desired correlation values
corr_array = np.zeros((len(r_d_array), 2))  # Pearson and Kendall corr coeffs
corr_diff = np.zeros((len(r_d_array), 1))  # difference between P and K

# %%
# Calculations
for i in range(len(r_d_array)):
    # Generate data
    x = rng.integers(low=1, high=n_cat+1, size=n_values)
    y = (x*r_d_array[i] + rng.integers(1, n_cat+1, n_values)
         * np.sqrt(1 - r_d_array[i]**2))
    # Let's shift the values along y so that they lie in the interval [1, 10]
    y = scale_to_interval(y, [1, n_cat])
    # Calculate and store Pearson and Kendall correlation coefficients
    corr_array[i, 0] = stats.pearsonr(x, y)[0]
    corr_array[i, 1] = stats.kendalltau(x, y)[0]
    corr_diff[i] = np.abs(corr_array[i, 0] - corr_array[i, 1])

# %%
# Visualization
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 6), layout='constrained')
fig.suptitle('Analysis of the difference between Pearson and Kendall '
             + 'correlation coefficients', fontsize=16)

ax1.plot([-1, 1], [-1, 1], 'r', linewidth=2)
ax1.plot(corr_array[:, 0], corr_array[:, 1], '*k', markerfacecolor='white',
         markersize=10)
ax1.set_title('Pearson vs. Kendall', fontsize=14)
ax1.set_xlabel('Person coefficient', fontsize=12)
ax1.set_ylabel('Kendall coefficient', fontsize=12)
ax1.grid()

ax2.plot(r_d_array, corr_diff, 'sk', markerfacecolor='blue',
         markersize=10)
ax2.set_title('The difference between computed coefficients',
              fontsize=13)
ax2.set_xlabel('Imposed correlation value', fontsize=12)
ax2.set_ylabel('Difference', fontsize=12)
ax2.grid()

'''
One conclusion we can draw here, in context of the goal of this exercise ("does K vs. P matter?"),
is that it doesn't matter at the "extremes" of very strong (close to |1|)
or very weak (close to 0) relationships.
But using the correct formula does matter for moderate relationships
(which will be the case in most real-world applications).
'''
