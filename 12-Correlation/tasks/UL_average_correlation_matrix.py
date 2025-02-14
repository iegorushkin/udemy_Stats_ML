# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:17:28 2021

@author: Igor
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# simulation parameteres
N_exp = 21  # number of experiments
N_ch = 20  # number of channels
N_time = 1000  # number of time points in each channel
time = np.linspace(0, 8*np.pi, N_time)  # time vector

# relationship across channels (imposing covariance)
# Задаём коэффициент, который:
# * равен нулю при x = 0, pi, 2pi
# * положителен при x = (0, pi)
# * отрицателен при x = (pi, 2pi)
chanrel = np.sin(np.linspace(0, 2*np.pi, N_ch))

# noise levels
noise_lvl = np.linspace(0.1, 5, N_exp)
# noise_lvl = np.linspace(0.1, 20, N_exp)  # very high noise
# %%
# generate data
# трехмерная матрица, в которой:
# - на нулевой оси откладываются каналы
# - на первой оси откладываются времена
# - на второй оси откладываются результаты различных экспериментов
data = np.zeros((N_ch, N_time, N_exp))

for i in range(N_exp):
    for j in range(N_ch):
        data[j, :, i] = (np.sin(time)*chanrel[j]
                         + noise_lvl[i]*rng.standard_normal(size=N_time))
#%%
# visualize the first and the last experiments
fig, ax = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
fig.suptitle('Visualizations of the experiments with different noise levels',
             fontsize=14,) # y=0.975)
# fig.subplots_adjust(hspace=0.3)

# First row
# plot each channel one above the other
for j in range(N_ch):
    ax[0, 0].plot(time, data[j, :, 0] + j*2)
    ax[0, 1].plot(time, data[j, :, -1] + j*150)
# make first row pretty
ax[0, 0].set(xlim=[0, np.max(time)], xlabel='Time, a.u.', ylabel='Channel',
             title=f"Experiment with random noise at level {noise_lvl[0]}")
ax[0, 0].set_yticks([0, 8, 18, 28, 38])  # Задаем позиции меток на оси Y
ax[0, 0].set_yticklabels(['1', '5', '10', '15', '20'])  # Задаем текст меток
ax[0, 1].set(xlim=[0, np.max(time)], xlabel='Time, a.u.', ylabel='Channel',
             title=f"Experiment with random noise at level {noise_lvl[-1]}")
ax[0, 1].set_yticks([0, 600, 1350, 2100, 2850])  # Задаем позиции меток на оси Y
ax[0, 1].set_yticklabels(['1', '5', '10', '15', '20'])  # Задаем текст меток
# Second row
cb1 = ax[1, 0].imshow(data[:, :, 0], aspect='auto', vmin=-1.5, vmax=1.5,
                      extent=[time[0], time[-1], 0, N_ch], origin='lower',
                      cmap='turbo')
ax[1, 0].set(xlabel='Time, a.u.', ylabel='Channel',
             yticks=[0, 4, 9, 14, 19], yticklabels=['1', '5', '10', '15', '20'])
fig.colorbar(cb1, ax=ax[1, 0], orientation='horizontal', label='Amplitude, a.u.',
             aspect=40)

cb2 = ax[1, 1].imshow(data[:, :, 19], aspect='auto', vmin=-7, vmax=7,
                      extent=[time[0], time[-1], 0, N_ch], origin='lower',
                      cmap='turbo')
ax[1, 1].set(xlabel='Time, a.u.', ylabel='Channel',
             yticks=[0, 4, 9, 14, 19], yticklabels=['1', '5', '10', '15', '20'])
fig.colorbar(cb2, ax=ax[1, 1], orientation='horizontal', label='Amplitude, a.u.',
             aspect=40)
# %%
# №1 Average all of the data and compute 1 correlation matrix
data_avg = np.mean(data, axis=2)
corr_matrix_1 = np.corrcoef(data_avg)

# №2 Compute the correlation matrix for each experiment and then average them
corr_matrix_temp = np.zeros((N_ch, N_ch, N_exp))
corr_matrix_2 = np.zeros((N_ch, N_ch))

for i in range(N_exp):
    corr_matrix_temp[:, :, i] = np.corrcoef(data[:, :, i])
corr_matrix_2 = np.mean(corr_matrix_temp, axis=2)
#%%
# №3 Visualize these results
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Correlation matrices obtained in 2 different ways', fontsize=14,
             y=0.98)
fig.subplots_adjust(hspace=0.3)

cb1 = ax[0].imshow(corr_matrix_1[:, :], aspect='equal', vmin=-.8, vmax=.8,
                   origin='upper', cmap='turbo')
ax[0].set(xticks=[0, 4, 9, 14, 19], xticklabels=['1', '5', '10', '15', '20'],
          yticks=[0, 4, 9, 14, 19], yticklabels=['1', '5', '10', '15', '20'],
          xlabel="Channel", ylabel="Channel",
          title='Correlation matrix of averaged data')
fig.colorbar(cb1, ax=ax[0], orientation='horizontal', label='Amplitude, a.u.',
             aspect=40)

cb2 = ax[1].imshow(corr_matrix_2[:, :], aspect='equal', vmin=-.8, vmax=.8,
                   origin='upper', cmap='turbo')
ax[1].set(xticks=[0, 4, 9, 14, 19], xticklabels=['1', '5', '10', '15', '20'],
          yticks=[0, 4, 9, 14, 19], yticklabels=['1', '5', '10', '15', '20'],
          xlabel="Channel", ylabel="Channel",
          title='Average of correlation matrces')
fig.colorbar(cb2, ax=ax[1], orientation='horizontal', label='Amplitude, a.u.',
             aspect=40)
'''
Take home message: 
Averaging the data before computing the correlation, in general,
allows to lower the effect of random noise.

Let's assume the noise is uncorrelated between x and y.
Then the effect of adding noise will increase the variances of x and y,
which makes the denominator term of the correlation coefficient larger,
which makes the correlation itself smaller.
Thus, averaging multiple correlation coefficients will make
the average correlation small.

If the data are averaged first,
then the noise cancels over the averaging process
(assuming that the noise is distributed around 0),
leading to cleaner data, which would produce a larger correlation coefficient.

This interpretation, however, can be complicated in certain datasets,
in which the averaging removes the correlated patterns between the variables.
This happens sometimes in time series data that have time-lags.

В решениях студентов с Udemy зачастую используется чертовский высокоамплитудный
шум, из-за которого выходит, что среднее корреляционных матриц даёт более или
менее диагональную матрицу, а корреляционная матрица усреднённых по экспериментам
каналов - нет.

Такой результат противоречит рассуждениям выше, но не является физичным
- амплитуда шума на порядок выше амплитуды полезного сигнала.
Объясняется он тем, что усреднение каналов (в том числе тех, что состоят из
высокоамплитудного шума) за все эксперименты и корреляция их друг с другом даёт
практически нулевой коэффициент корреляции.

А вот расчёт отдельных матриц корреляции с их последующим усреднением даёт
"блоковую" картину, которую мы и ожидаем увидеть, пусть блоки и обладают низкой
амплитудой [для сильного шума вне диагонали соответствующей
корреляционной матрицы ~нули, но при низком шуме там находятся значения
отличные от нуля; усредняем матрицы, получаем матрицу, у которой за счёт деления
на количество экспериментов имеются низкоамплитудные блоки].
'''
