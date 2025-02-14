# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:53:47 2021

@author: Igor
"""
# import libraries
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
# %%
# Create the dataset

rng = np.random.default_rng()

# The first part of the dataset - normal distribution
d1_lenght = 144
d1_values = rng.normal(loc=2.5, scale=0.75, size=d1_lenght)

# The second part of the dataset - (log)normal distribution
d2_lenght = 187
d2_values = rng.normal(loc=2.2, scale=1.3, size=d2_lenght)
# d2_values = rng.lognormal(mean=0.5, sigma=1.62, size=d2_lenght)

# Calculate the observed "t"-value
d1_mean = np.mean(d1_values)
d2_mean = np.mean(d2_values)
obs_val = d1_mean - d2_mean

# Concat both parts
data = np.hstack((d1_values, d2_values))
d_lenght = len(data)

# Plot for comparison purposes
with plt.style.context('seaborn'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('Distribution of 2 parts of the dataset', fontsize=16, y=0.975)
    fig.subplots_adjust(hspace=0.25)

    ax[0].hist(d1_values, bins=20, color='b', edgecolor='k')
    ax[0].plot([np.mean(d1_values), np.mean(d1_values)], [0, 25], 'r--', lw=3,
               label=(r'$\bar{x}$ ='+f'{np.round(d1_mean, 3)}'))
    ax[0].set_title(f"Normal distribution, n={d1_lenght}, mu=2.5, std=0.75",
                    fontsize=13)
    ax[0].set(xlabel='Values', ylabel='Counts')
    ax[0].legend(frameon=True, framealpha=1, facecolor='white')
    
    
    ax[1].hist(d2_values, bins=20, color='r', edgecolor='k')
    ax[1].plot([np.mean(d2_values), np.mean(d2_values)], [0, 25], 'b--', lw=3,
               label=(r'$\bar{y}$ ='+f'{np.round(d2_mean, 3)}'))
    ax[1].set_title(f"Normal distribution, n={d2_lenght}, mu=2.2, std=1.3",
                    fontsize=13)
    ax[1].set(xlabel='Values', ylabel='Counts')
    ax[1].legend(frameon=True, framealpha=1, facecolor='white')
fig
#%%
n_exp = range(100, 5001, 50)  # the number of permutations per experiment
Z_values = np.zeros(len(n_exp))  # array for the calculated Z-scores

# For each loop we need the number of permutations per experiment (value),
# as well as the count of this number in the overall range (index)
for index, value in enumerate(n_exp):
    exp_results = np.zeros(value)  # the values of the null hypothesis are stored here
    # Carry out an experiment consisting of 'value' different permutations.
    for i in range(value):
        data_groups = np.zeros(d_lenght)
        permutation = rng.permutation(d_lenght)  # without replacement
        # Splitting data into two random groups
        data_groups[permutation < d1_lenght] = 1
        data_groups[permutation >= d1_lenght] = 2
        # Calculation of the difference between the mean values for the two groups.
        # This value is a "t"-statistic
        # and is used to construct an empirical null hypothesis.
        exp_results[i] = (np.mean(data[data_groups == 1])
                          - np.mean(data[data_groups == 2]))
    # Evaluation of the statistical significance of the observed "t"-statistic
    # relative to the empirical null hypothesis.
    cur_Z = (obs_val - np.mean(exp_results)) / np.std(exp_results, ddof=1)
    # Saving the score to an array.
    Z_values[index] = cur_Z

'''
Смысл Z-перехода в том, что мы не значем, какая вероятность соответствует
наблюдённому значению, ведь распределение нулевой гипотезы, построенное из
имеющихся данных, не непрерывно.
Мы видим / знаем / предполагаем, что нулевая гипотеза имеет распределение,
близкое к нормальному.
С учётом этого предположения, по формуле выполняем Z-переход к стандартному нормальному
распределению, в котором нашему наблюденному значению ставится
в соответствие какой-то P-value.
'''
# %%
with plt.style.context('ggplot'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)

    ax[0].plot(n_exp, Z_values, 'ok', markerfacecolor='white',
               markersize=12)
    ax[0].set(xlabel='Permutations per experiment', ylabel='Z-score',
              title="Effect of the number of permutations on Z-score")

    ax[1].hist(Z_values, bins=30, color='w', edgecolor='k')
    ax[1].set(xlabel='Z-score', ylabel='Count', 
              title="Z-score distribution from various experiments")
    # Одно из эмперических распределений нулевой гипотезы
    # ax[1].hist(exp_results, bins=30, color='w', edgecolor='k') 
    # Место наблюденного значения на этом распределении  
    # ax[1].plot([obs_val, obs_val], [0, 500], linewidth=10)
fig
# %%
# The mean of Z_values gives the best estimate of the result of the permutation tests
# But is this result good or bad?

Z_avg = np.mean(Z_values)
p_avg = 1 - stats.norm.cdf(np.abs(Z_avg))

ax[1].plot([Z_avg, Z_avg], [0, 20], color='red', linewidth=3, linestyle="--",
           label="mean")
ax[1].legend(frameon=True, framealpha=1, facecolor='white')
fig

print(f"Mean Z-score: {Z_avg}" +
      f"\nCorresponding P-value: {p_avg}")

#%%
'''
P-значение: вероятность наблюденной t-статистики при условии что наборы данных
распределены по одному закону и их средние значения равны.
Маленькое P-значение -> доказательство того, что нулевая гипотеза ошибочна
'''
x = np.linspace(-3, 3, 1000)
with plt.style.context('ggplot'):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Visualization of the permutation tests result',
                 fontsize=16, y=0.95)

    ax.plot(x, stats.norm.pdf(x), 'b', lw=3, label='Standard normal distr.')
    ax.plot([Z_avg, Z_avg], [0, 0.45], 'r', lw=3,
            label='Estimation of the Z-score')
    ax.plot([stats.norm.ppf(0.95), stats.norm.ppf(0.95)], [0, 0.45],
            'm--', lw=2, label='Significance threshold')
    ax.set(xlabel='Z-score, stds', ylabel='Probabilities, %')
    ax.legend(frameon=True, framealpha=1, facecolor='white')
fig
