# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:22:24 2021

@author: Igor
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

rng = np.random.default_rng()
# Simulate some data
N = 100
r_d = 0.45  # Desired Pearson correlation between x and y
x = rng.standard_normal(size=N)
x_mean = np.mean(x)
# imposing correlation with x on y
y = x*r_d + rng.standard_normal(size=N)*np.sqrt(1 - r_d**2)  
y_mean = np.mean(y)

# Plot these data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'ok', markersize=7, markerfacecolor='white')
# ax.set_title('y vs. x', fontsize=14)
ax.set_xlabel('x values', fontsize=12)
ax.set_ylabel('y values', fontsize=12)
ax.grid()

# %%
# Compute Pearson's correlation coefficient ('by hand')
r_p = (np.sum((x - x_mean)*(y - y_mean))
       / np.sqrt(np.sum((x - x_mean)**2)*np.sum((y - y_mean)**2)))
ax.set_title(f'y vs. x, \ncorrelations: imposed = {r_d}, observed = {np.round(r_p, 2)}',
             fontsize=14)

# %%
# Now for the confidence interval (via bootstrapping)
perm_s = 76  # size of permutated samples
perm_n = 1000  # number of permutations
# array for storing the required parameter from each permutation sample
r_p_array = np.zeros(perm_n)

# resample perm_n-times, compute and store Pearson correlation coefficients
for i in range(perm_n):
    # for the method to work, must use the same indices for x and y;
    # not different random values from x and y.
    indices_cur = rng.choice(a=np.arange(N), size=perm_s)  # array of size N
    # Select the x and y values corresponding to indices_cur and center them  
    x_cur = x[indices_cur] - np.mean(x[indices_cur])   
    y_cur = y[indices_cur] - np.mean(y[indices_cur])  

    r_p_array[i] = (np.sum(x_cur*y_cur)
                    / np.sqrt(np.sum(x_cur**2)*np.sum(y_cur**2)))

# find 95% - CI boundaries
ci_bounds = np.percentile(r_p_array, [2.5, 97.5])

# %%
# Visualize the results
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Confidence interval (via bootstrapping) of Pearson's r",
             fontsize=16, y=0.96)

# distribution
n, _, _ = ax.hist(r_p_array, bins=40, color='b', edgecolor='k',
                  label='empirical dist.')
# green patch
temp = [[ci_bounds[0], 0], [ci_bounds[1], 0],
        [ci_bounds[1], np.max(n)], [ci_bounds[0], np.max(n)]]
p = Polygon(temp, facecolor='m', alpha=0.3, label='95% CI')
ax.add_patch(p)
# lines
ax.plot([r_d, r_d], [0, np.max(n)], 'r-', linewidth=3,
        label=f'imposed r = {np.round(np.mean(r_d), 2)}')
ax.plot([r_p, r_p], [0, np.max(n)], 'g--', linewidth=3,
        label=f'sample r = {np.round(np.mean(r_p), 2)}')
ax.plot([np.mean(r_p_array), np.mean(r_p_array)], [0, np.max(n)],
        'y-.', linewidth=3, label=('estimation of the imposed r '+
                                  f'= {np.round(np.mean(r_p_array), 2)}'))
# axis customization 
ax.set_xlabel('r value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.tick_params(axis='both', which='both', labelsize=12)
ax.legend(framealpha=1, frameon=True, loc='upper left', fontsize=10)

# %%
# Correlation coefficient vs. lenght of the confidence interval

# number of values drawn from a Normal distribution   
N = 100
# size of permutated samples  
N_p = 93
# number of permutations
perm_n = 1000  
# array of the desired Pearson correlations between x and y
r_d_array = np.linspace(-1, 1, 51)
#  array with widths of the corresponding 95% confidence intervals
ci_array = np.zeros(len(r_d_array))

for i in range(len(r_d_array)):
    x = rng.standard_normal(size=N)
    # imposing correlation with x on y
    y = (x*r_d_array[i]
         + rng.standard_normal(size=N)*np.sqrt(1 - r_d_array[i]**2))
    r_p = stats.pearsonr(x, y)[0]
    # array for storing the required parameter from each permutation sample
    r_p_array = np.zeros(perm_n)

    # resample perm_n-times, compute and store Pearson correlation coefficients
    for j in range(perm_n):
        # for the method to work, must use the same indices for x and y;
        # not different random values from x and y.
        indices_cur = rng.choice(a=np.arange(N), size=perm_s)
        # Select the x and y values corresponding to indices_cur and center them 
        x_cur = x[indices_cur]
        y_cur = y[indices_cur]
        r_p_array[j] = stats.pearsonr(x_cur, y_cur)[0]

    # find 95% - CI boundaries
    ci_bounds = np.percentile(r_p_array, [2.5, 97.5])
    # save the length of the current CI
    ci_array[i] = ci_bounds[1] - ci_bounds[0]

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Relationship between the correlation strength \nand the length" +
             " of the 95% confidence intervals",
             fontsize=14, y=0.96)
ax.plot(r_d_array, ci_array, 'ok', markersize=7, markerfacecolor='white')
ax.set_xlabel('Correlation coefficient', fontsize=12)
ax.set_ylabel('Lenght of the 95% CI', fontsize=12)
'''
Вывод: сильнее корреляция (неважно, с каким знаком) - уже доверительный интервал,
полученный с помощью bootstraping 
'''