# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:10:17 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Parameters
mu = 5
std = 0.5
n = 5000

# Generate data
data = std*np.random.randn(n) + mu
log_data = np.exp(data)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
fig.subplots_adjust(hspace=0.3)
fig.suptitle(f'Lognormal distribution\n $m$={mu}, std={std}, n={n}', y=0.95)

ax[0].plot(log_data, '.')
ax[0].set(ylabel='sample values', xlabel='sample numbers')

ax[1].hist(log_data, bins=30)
ax[1].set(ylabel='counts', xlabel='bins')

mu_log = np.mean(np.log(log_data))
std_log = np.std(np.log(log_data))
print(f"mu_log={mu_log}, std_log={std_log}")
