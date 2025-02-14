# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:08:02 2021

@author: Igor
"""
import numpy as np
import matplotlib.pyplot as plt

# data
N = 1250
rng = np.random.default_rng()
data = 5.5*rng.standard_normal(size=N) + rng.uniform(size=N)
random_shift = 1 + rng.standard_normal(size=N)/20

#%%
# [0, 1] scaling
data_min = np.min(data)
data_max = np.max(data)

data_s1 = (data - data_min) / (data_max - data_min)

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
fig.subplots_adjust(wspace=0.4)
ax[0].plot(random_shift, data, '.k')
ax[0].set(xticks=[], ylabel='Values', title='Unscaled dataset')
ax[1].plot(random_shift, data_s1, '.r')
ax[1].set(xticks=[], ylabel='Values', title='Scaled dataset ([0, 1])')

#%%
# arbitrary scaling
a = 7.5
b = 28

data_s2 = a + (b - a)*data_s1
ax[2].plot(random_shift, data_s2, '.c')
ax[2].set(xticks=[], ylabel='Values', title=f'Scaled dataset (a={a}, b={b})')

#%%
# reverse scaling
data_s3 = data_min + (data_max - data_min)*(data_s2 - a)/(b - a)

fig, ax = plt.subplots()
ax.plot(data, data_s3, '*')
ax.set(title='Correlation between the original and reversed-scaled data',
       xlabel='Original', ylabel='Scaled')
ax.grid()
