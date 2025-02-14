# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:20:15 2021

@author: Igor
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

N = 400

rng = np.random.default_rng()
data1 = rng.standard_normal(size=N)
data2 = rng.uniform(low=0, high=1, size=N)**3
#list of 2 elements, which are numpy arrays
data = [data1, data2]
#%%
for i in range(len(data)):
    data_zscore = (data[i] - np.mean(data[i])) / np.std(data[i])
    data_zscore_mod = (stats.norm.ppf(0.75)*(data[i] - np.median(data[i]))
                       / stats.median_abs_deviation(data[i]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Different types of the data normalization', y=0.94,
                 fontsize=12)
    fig.subplots_adjust(wspace=0.3)

    ax[0].plot(data_zscore, '*k', markerfacecolor='white', markersize=12,
               label='z-score')
    ax[0].plot(data_zscore_mod, '.k', markerfacecolor='red', markersize=12,
               label='mod. z-score')
    ax[0].set(xlabel='indices', ylabel='vales',)
    ax[0].grid()
    ax[0].legend(frameon=True, framealpha=1, loc='upper right', fontsize=10)

    ax[1].plot(data_zscore, data_zscore_mod, 'ok', markerfacecolor='b',
               markersize=12)
    ax[1].set(xlabel='z-score', ylabel='mod. z-score',)
    ax[1].grid()

'''
If we have a normally distributed data (i = 0),
the mean value and median value are close to each other,
so in the end the Z-score and modified Z-score are likely to be
distributed more or less the same.

But in the case of non-normally distributed data (i = 1) mean value
is not reliable, as it is not describing the central tendency of the dataset.
In this case median value is more precise and better describes
the central tendency, that is why we have a different distribution of z-score
and modified z-score.
'''
