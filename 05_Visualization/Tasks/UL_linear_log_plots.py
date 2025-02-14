# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 23:13:24 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np

# What styles are available?
## plt.style.available

# setup
n = 1000
x = np.linspace(0.0001, 10, n)
y1 = x
y2 = np.exp(x)

# plot
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    fig.set_layout_engine(layout='constrained')

    ax[0].plot(x, y1, lw=2, c='k', label='$y = x$',)
    ax[0].plot(x, y2, lw=2, c='c', label='$y = e^x$')
    ax[0].legend(loc='upper left', frameon=True, framealpha=1, fontsize=12)
    ax[0].set(title='Linear scale', xlabel='x', ylabel='y', xlim=[0, 10],
              ylim=[0.001, 100])

    ax[1].plot(x, y1, lw=2, c='k', label='$y = x$',)
    ax[1].plot(x, y2, lw=2, c='c', label='$y = e^x$')
    ax[1].legend(loc='lower right', frameon=True, framealpha=1, fontsize=12)
    ax[1].set(title='Y-logarithmic scale', xlabel='x', ylabel='y',
              xlim=[0, 10], ylim=[0.001, 100], yscale='log')
