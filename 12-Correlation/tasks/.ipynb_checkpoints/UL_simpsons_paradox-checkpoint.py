# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:57:20 2021

@author: Igor
"""
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

rng = np.random.default_rng()  # random generator
n_sg = 3  # number of subgroups in the data
n_sg_values = 100   # number of data points in each subgroup
r_sg = [-0.6, -0.4, -0.75]  # correlation coefficients imposed on the subgroups
c = 'rgb'  # subgroup's colors
offsets = [0, 1.5, 3]  # mean offsets
# Arrays for storing the complete set of x and y values.
all_x = np.array([])
all_y = np.array([])

# %%
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Simpson's paradox illustration", fontsize=14, y=0.96)

for i in range(n_sg):
    # Creation of data for the current subgroup,
    # taking into account the desired correlation coefficient and offset value
    x = rng.standard_normal(size=n_sg_values)/2 + offsets[i]
    y = (np.sqrt(1-r_sg[i]**2) * rng.standard_normal(size=n_sg_values)/2
         + x*r_sg[i] + offsets[i])
    # Calculation of the corr. coeff directly from the current subgroup data
    r_p, p_value = pearsonr(x, y)
    # Append the current subgroup data to arrays
    all_x = np.append(all_x, x)
    all_y = np.append(all_y, y)
    # Plot the subgroup and its linear correlation.
    ax.plot(x, y, 'o', color=c[i], markersize=10,
            label=f"r={r_p:.2f}")
    x_corr = np.array([np.min(x), np.max(x)])
    y_corr = x_corr*r_p + offsets[i]
    ax.plot(x_corr, y_corr, color=c[i], alpha=0.5, linewidth=2)

# %%
# Corr. coeff of the whole dataset
r_p, p_value = pearsonr(all_x, all_y)
# Add this linear correlation to the plot
x_corr = np.array([np.min(all_x), np.max(all_x)])
y_corr = x_corr*r_p + offsets[0]
ax.plot(x_corr, y_corr, 'k--', linewidth=2)
# Make the plot more beatiful
ax.set_title(f'Result across all data: r={r_p:.2f}',
             fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(framealpha=1, frameon=True, fontsize=10,
          title='individual subgroups')

# Take home message:
# A combination of 3 groups, each of which has a negative correlation,
# can have a POSITIVE correlation.