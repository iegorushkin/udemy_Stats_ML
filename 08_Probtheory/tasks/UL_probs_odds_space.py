# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 18:05:38 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np

n = range(1, 11)
m = range(1, 11)

x, y = np.meshgrid(m, n)
p = (1 / (1 + y/x))

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Probabilities of odd-space', y=0.93)

pic = ax.imshow(p, extent=[1, 10, 1, 10], origin='lower', cmap='hot',
                vmin=0, vmax=1)
ax.set(xlabel='m', ylabel='n',)
fig.colorbar(pic, ax=ax, label='Probability', aspect=20, shrink=1,)
             #boundaries=np.linspace(0, 1, 11), extend='both')
