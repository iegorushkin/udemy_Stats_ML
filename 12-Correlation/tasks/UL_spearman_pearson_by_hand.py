# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:51:33 2021

@author: Igor
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# simulate some data
rng = np.random.default_rng()
N = 1000
r_d = 0.47  # desired Pearson correlation between x and y
x = rng.standard_normal(size=N)
y = x*r_d + rng.standard_normal(size=N)*np.sqrt(1 - r_d**2)
x[-3] = 100

# data means
x_mean = np.mean(x)
y_mean = np.mean(y)

# plot these data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'ok', markersize=7, markerfacecolor='white')
# ax.plot([-3, 3], [-3*r_d, 3*r_d])  # interesting to observe
ax.set_title(f'y vs. x, \nImposed correlation is {r_d}', fontsize=14)
ax.set_xlabel('x values', fontsize=12)
ax.set_ylabel('y values', fontsize=12)
ax.grid()
# %%
# compute Pearson's correlation coefficient ('by hand')
r_p = (np.sum((x - x_mean)*(y - y_mean))
       / np.sqrt(np.sum((x - x_mean)**2)*np.sum((y - y_mean)**2)))
# %%
# compute Spearman's correlation coefficient ('by hand')

# this instead of using stats.rankdata
def rank_transform(data):
    data_copy = data.copy()
    data_len = len(data)
    data_rank = np.zeros(len(data), dtype=int)  # output, rank-transformed vector

    # loop through every index value of data     
    for i in range(data_len):
        cur_position, cur_step = 0, 1 
        # The main idea is that we compare the value of the data
        # in the cur_position + cur_step cell with the value in cur_position.
        # The sum of these indices can't be > the length of the dataset
        while cur_position + cur_step < data_len:
            # If the value in the cell with index cur_position + cur_step is
            # less than the value in the cell with index cur_position,
            # we shift cur_position and reset the step.
            if (data_copy[cur_position + cur_step] < data_copy[cur_position]
                or np.isnan(data_copy[cur_position])): 
                # ^ change position if data_copy[cur_position] is NaN
                cur_position = cur_position + cur_step
                cur_step = 1
            # If the value in the cell with index cur_position + cur_step is
            # greater than (or equal to) the value in the cell with index cur_position,
            # we increase the step and perform a new comparison.
            elif (data_copy[cur_position + cur_step] >= data_copy[cur_position]
                  or np.isnan(data_copy[cur_position + cur_step])):
                # ^ increase step if data_copy[cur_position + cur_step] is NaN
                cur_step += 1
        # Having found the index of the minimum element in the data,
        # we declare the value corresponding to it as NaN.
        # This allows us to impose additional conditions in the if-else loop,
        # which ensure finding successively increasing values in the data and their indices.
        # NOTE: Any comparison with NaN gives False        
        data_copy[cur_position] = np.nan
        # The result of current iteration of the rank-transform.
        data_rank[cur_position] = int(i)

    return data_rank

x_r = rank_transform(x)
y_r = rank_transform(y)
x_r_mean = np.mean(x_r)
y_r_mean = np.mean(y_r)

r_s = (np.sum((x_r - x_r_mean)*(y_r - y_r_mean))
       / np.sqrt(np.sum((x_r - x_r_mean)**2)*np.sum((y_r - y_r_mean)**2)))
# %%
# let's add the coefficients to the plot
y_p = r_p*np.sort(x)
y_s = r_s*np.sort(x)
ax.plot([np.min(x), np.max(x)], [r_p*np.min(x), r_p*np.max(x)], '--',
        linewidth=2, label=f'r (Pearson) = {np.round(r_p, 2)}')
ax.plot([np.min(x), np.max(x)], [r_s*np.min(x), r_s*np.max(x)], ':',
        linewidth=2, label=f'r (Spearman) = {np.round(r_s, 2)}')
ax.legend(framealpha=1, frameon=True, loc='lower right', fontsize=12)
# %%
# compare the results with Pearson and Spearman from scipy built-in functions
corr_p = stats.pearsonr(x, y)[0]
corr_s = stats.spearmanr(x, y)[0]
print(r_p, corr_p)
print(r_s, corr_s)
print('Difference between correlations calculated in different ways:')
print(f"Pearson: {np.abs(r_p - corr_p)}")
print(f"Spearman: {np.abs(r_s - corr_s)}")
print('Success!')
