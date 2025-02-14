# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 18:47:20 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np

# generate data
n = 1000
rng = np.random.default_rng()
data = rng.standard_normal(n)

# create counts and bins
bin_counts, bin_edges = np.histogram(data, 30)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=150)
fig.set_layout_engine(layout='constrained')

# plot histogram in counts
ax[0].hist(bin_edges[:-1], bin_edges, weights=bin_counts)
ax[0].set(title='Histogram (counts) from np.histogram', xlim=[-3.5, 3.5],
          xlabel='bins', ylabel='counts')

# convert counts to percents
# for-loop
# phist = np.zeros(chist.shape)
# for i in range(30):
#     phist[i] = (chist[i] / n) * 100

# more numpying way
bin_perc = (bin_counts / n) * 100

# plot histogram in percents
ax[1].hist(bin_edges[:-1], bin_edges, weights=bin_perc)
ax[1].set(title='Histogram (%) from np.histogram', xlim=[-3.5, 3.5],
          xlabel='bins', ylabel='%')
# showing off
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax[1].text(0.02, 0.85, f'Sum of all bins is {np.round(np.sum(bin_perc), 2)}',
           transform=ax[1].transAxes, bbox=props)
plt.show()
