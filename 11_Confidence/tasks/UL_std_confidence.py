import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Polygon

rng = np.random.default_rng()

# simulate some population
pop_n = int(1e6)
pop_data = (4*rng.normal(loc=2, scale=5, size=pop_n))**2
# population parameter
pop_std = np.std(pop_data)

# %%
#  visualize it
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle('Simulated population', fontsize=16, y=0.975)
fig.subplots_adjust(hspace=0.4)

ax[0].plot(np.arange(pop_n)[::1000], pop_data[::1000], '*k', )
ax[0].set_xlim([0, pop_n])
ax[0].set_xlabel('Data index', fontsize=12)
ax[0].set_ylabel('Data value', fontsize=12)
ax[0].set_title('What does the data look like?', fontsize=14)
ax[0].tick_params(axis='both', which='both', labelsize=10)

ax[1].hist(pop_data, bins=40, color='b', edgecolor='k')
ax[1].set_xlim([0, 12000])
ax[1].set_xlabel('Data value', fontsize=12)
ax[1].set_ylabel('Counts', fontsize=12)
ax[1].set_title('The data histogram', fontsize=14)
ax[1].tick_params(axis='both', which='both', labelsize=10)
# %%
# Draw some sample from the population
sample_n = 100  # N
sample_data = rng.choice(a=pop_data, size=sample_n)  # data itself
sample_std = np.std(sample_data, ddof=1)  # sample parameter
# %%
# Confidence interval via bootstrapping
perm_n = 5000  # number of permutations
# array for storing the required parameter from each permutation sample
stds_array = np.zeros(perm_n)

# resample perm_n-times, compute and store stds
for i in range(perm_n):
    # note - not the same array length as sample_n
    stds_array[i] = np.std(rng.choice(a=sample_data, size=69), ddof=1)

# find CI boundaries
ci_bounds = np.percentile(stds_array, [2.5, 97.5], method="linear")
# There are a lot of options for method, but I didn't dive into them.
# So, basic linear interpolation!
#%%
# visualize
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Confidence interval via bootstrapping\n(resampling)',
             fontsize=16, y=0.975)

# distribution
n, _, _ = ax.hist(stds_array, bins=40, color='b', edgecolor='k',
                  label='Empirical dist.')
# green patch
# np.max(stds_array)
temp = [[ci_bounds[0], 0], [ci_bounds[1], 0],
        [ci_bounds[1], np.max(n)], [ci_bounds[0], np.max(n)]]
p = Polygon(temp, facecolor='m', alpha=0.3, label='95% CI region')
ax.add_patch(p)
# lines
ax.plot([pop_std, pop_std], [0, np.max(n)], 'r-', linewidth=3,
        label=f'True population std = {np.round(pop_std, 2)}')
ax.plot([sample_std, sample_std], [0, np.max(n)], 'g--', linewidth=2,
        label=f'Sample std = {np.round(sample_std, 2)}')
ax.plot([np.mean(stds_array), np.mean(stds_array)], [0, np.max(n)],
        'y-.', linewidth=2,
        label=f'Estimated population std = {np.round(np.mean(stds_array), 2)}')
# plot customization
ax.set_xlabel('Data value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.tick_params(axis='both', which='both', labelsize=10)
ax.legend(framealpha=1, frameon=True, loc='best', fontsize=12)
'''
Гипотетически, можно построить аналитический доверительный интервал
для среднеквадратичного отклонения. Формула есть по ссылке:
http://www.milefoot.com/math/stat/ci-variances.htm
Выглядит довольно легко, главная разница:
вместо т-распределения используется хи-распределение
'''
