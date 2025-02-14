# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:54:33 2021

@author: Igor
"""
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
# %%
# Expected values (means) of 2 unpaired datasets
ds1_mean = 1
ds2_mean = 1.2

# Sample size range of these 2 groups
sample_sizes = np.arange(10, 201)
# sample_sizes = range(10, 201)

# Number of experiments for each sample size
n_exp = 100

# Random generator
rng = np.random.default_rng()

# Arrays for storing the results of T-tests
t_values = np.zeros((len(sample_sizes), n_exp))
p_values = np.zeros((len(sample_sizes), n_exp))

# Arrays for storing the average t and p values after 100 experiments
t_means = np.zeros(len(sample_sizes))
p_means = np.zeros(len(sample_sizes))

# P-value threshold
pval_threshold = 0.05
# Array of t-values corresponding to the threshold
t_threshold = np.zeros(len(sample_sizes))
# %%
# Calculations
for i in range(len(sample_sizes)):
    # First of all, let's define a t-value corresponding to
    # the p-value threshold given the current sample size.
    df = 2*sample_sizes[i] - 2  # current degrees of freedom
    t_threshold[i] = stats.t.ppf(q=(pval_threshold), df=df)
    '''
    Что происходит в двух строчках кода выше?
    В 39 строке задаётся количество степеней свободы. Для каждой выборки оно
    определяется как n-1, а для двух выборок как (n1 - 1) + (n2 - 1).
    Очень сильно упрощая, количество степеней свободы -
    это число элементов выборки, которые можно свободно менять, зная среднее
    значение выборки.
    В 40 строке выполняется поиск т-статистики, соответствующей вероятности 5%.
    Percent point function (ppf) - функция, обратная к сumulative distribution function
    (cdf).
    В cdf по оси х отложены значения распределения,
    по оси у - сумма всех вероятностей левее этого значения. Соответственно,
    в ppf всё наоборот.
    Пихая в функцию искомый процент (например, 5%), на выходе
    получаем значение распределения, левее которого лежит 5% данных.
    
    По идее, в "двух-хвостом" ти-тесте нужно брать пороговые значения
    с обеих сторон распределения - то есть, 95% и 5% (в таком случае говорят,
    что пороговое значение равно 10%).
    '''
    for j in range(n_exp):
        # Draw random samples from a normal distribution
        # with the current mean and sample size
        ds1 = rng.normal(loc=ds1_mean, scale=1.0, size=sample_sizes[i])
        ds2 = rng.normal(loc=ds2_mean, scale=1.0, size=sample_sizes[i])
        # ds1 = np.random.normal(loc=ds1_mean, scale=1, size=sample_sizes[i])
        # ds2 = np.random.normal(loc=ds2_mean, scale=1, size=sample_sizes[i])

        # Calculate the T-test for the means of TWO INDEPENDENT samples.
        # If equal_var=True (default), perform a standard independent
        # 2 sample test that assumes equal population variances.
        t_values[i, j], p_values[i, j] = stats.ttest_ind(ds2, ds1,
                                                         alternative="greater")

# Averaging p-values and t-statistics for all experiments
t_means = np.mean(t_values, axis=1)
p_means = np.mean(p_values, axis=1)
# %%
# Visualization
fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
fig.suptitle('Effect of sample size on T-values' +
             f'\n(two sample t-tests with means {ds1_mean} and {ds2_mean})',
             fontsize=14)

# Some trickery for creating the correct legend
ax.plot(sample_sizes, t_values[:, :99],  c='0.75',)
ax.plot(sample_sizes, t_values[:, 99],  c='0.75',
        label='individual experiments')
ax.plot(sample_sizes, t_threshold, 'r--', lw=2)
ax.plot(sample_sizes, -t_threshold, 'r--', lw=2, label='P-value threshold')

ax.plot(sample_sizes, t_means, c='0', label=f'mean of {n_exp} experiments',
        linewidth=3)
# ax.plot(sample_sizes, p_means, c='m')  # apparently, not really correct to do it
ax.set(xlim=[10, 200], xlabel='Sample size', ylabel='T-values')
ax.legend()
