# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:24:27 2021

@author: Igor

Почему гистограмма аналитического распределения Гаусса
выглядит как черт знает что?
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# number of discretizations
N = 1001

x = np.linspace(-4, 4, N)
gausdist = stats.norm.pdf(x) # probability density function

# is this a probability distribution?
print(f'The sum of all values in this distribution is {np.sum(gausdist)}')
# try scaling by dx...
print('Calculating the Riman sum of the distribution...')
print(f'= {np.sum(gausdist*np.mean(np.diff(x)))}')

# plot data
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle('Analytic Gaussian (normal) distribution', y=0.95)

    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(x, gausdist)
    ax0.plot(x[::30], gausdist[::30], 'sr')
    ax0.set_xlim([np.min(x), np.max(x)])

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.hist(gausdist, 40, edgecolor='black', linewidth=1)

plt.show()

'''
Мы строим гистограмму значений gausdist, по которым рисуется гауссиан
А не гистограмму значений данных, имеющих распределение гаусса
'''
