# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 00:03:45 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np

# Number 1: all possible averages between two uniformly distributed integers
# from 1 to 100
x = np.arange(1, 101)
y = np.arange(1, 101)
means1 = np.zeros(len(x)*len(y))

for i in range(len(x)):
    for j in range(len(y)):
        means1[100*i + j] = (x[i] + y[j]) / 2

fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ax[0].hist(means1, bins='fd')
ax[0].set(ylabel='Counts', xlabel='Values',
          title=('Means obtained from two' +
                 '\nuniformly distributed integers from 1 to 100'))

# Number 2: Honest population and samples from it
population = np.random.randint(low=1, high=101, size=1000000)

n_samples = 10000
means2 = np.zeros(n_samples)

for i in range(n_samples):
    means2[i] = np.mean(np.random.choice(population, 2,)) # replace=False))

ax[1].hist(means2, bins='fd', color='red')
ax[1].set(ylabel='Counts', xlabel='Values',
          title=('Means obtained from \nuniformly distributed population'))
plt.tight_layout()
