# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 21:04:37 2021

@author: Igor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Brownian noise = cumulative sum of random +1 and -1
N = 1500
rng = np.random.default_rng()
data = np.cumsum(np.sign(rng.standard_normal(N)))

entropy = np.zeros(47)
for i, bins in enumerate(range(4, 51)):
    counts, edges = np.histogram(data, bins)
    entropy[i] = - np.sum(counts/N * np.log2(counts/N + 1e-16))

# Plot
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.4)

ax[0].plot(data, '-')
ax[0].set(title='Brownian noise', xlabel='index', ylabel='value')
ax[1].plot(range(4, 51), entropy, 'o-k')
ax[1].set(title='Relationship between entropy and the number of bins',
          xlabel='bins', ylabel='entropy')

# Bigger number of bins -> bigger entropy