# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 19:03:58 2021

@author: Igor
"""
# import libraries
import matplotlib.pyplot as plt
import numpy as np

# parameters
mu = 5
std = 1
N = 10001   # number of data points
nbins = 100  # number of histogram bins

# generate data
data = np.random.normal(mu, std, N)

# emperical mean and std
mean1 = np.mean(data)
std1 = np.std(data, ddof=1)

# histogram
counts, bin_edges = np.histogram(data, nbins)
x = np.zeros(len(bin_edges-1))
x = (bin_edges[1:] + bin_edges[:-1])/2

# for plotting std-data chunck
ipeak = np.argmin((x-mean1)**2)
istd_l = np.argmin((x[:ipeak+1]-(mean1-std1))**2)
istd_r = ipeak + np.argmin((x[ipeak:]-(mean1+std1))**2)
x1 = np.concatenate(([x[istd_l]], x[istd_l:istd_r+1], [x[istd_r]]))
y1 = np.concatenate(([0], counts[istd_l:istd_r+1], [0]))

plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
# plot
ax.plot(x, counts, lw=2)
ax.fill_between(x1, 0, y1, alpha=0.7, facecolor='c', label='std')
ax.plot([x[ipeak], x[ipeak]], [0, counts[ipeak]], 'r--', lw=2, label='mean')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.set_title('Normal distribution', fontsize=14)
ax.set_xlabel('values', fontsize=12)
ax.set_ylabel('counts', fontsize=12)
ax.legend(framealpha=1, frameon=True, loc='upper right', fontsize=12)
