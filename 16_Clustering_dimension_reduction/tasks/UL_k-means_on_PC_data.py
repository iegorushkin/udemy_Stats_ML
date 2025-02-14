# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 21:54:34 2022

@author: Igor
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# load data
spikes = np.loadtxt('spikes.csv', delimiter=',')
# %%
# let's see it!
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                       gridspec_kw={'height_ratios': [0.5, 2]},)
                       #constrained_layout=True)

ax[0].plot(np.mean(spikes, axis=0))
ax[0].set(ylabel='Amplitude', title='Average of all spikes')

cb = ax[1].imshow(spikes, aspect='auto', cmap='jet')

ax[1].set(ylabel='Spike', xlabel='Time points', title='Individual spikes')
fig.colorbar(cb, ax=ax[1], orientation='horizontal',
             label='Amplitude, a.u.', shrink=0.6, aspect=35, pad=0.17)
# %%
# PCA using scikitlearn's function
'''
По сути, в основе PCA лежит процесс нахождения собственных значений и
собствеенных векторов, отвечающих этим значениям.
Чем больше собственное значение - тем "главнее" отвечающий ей собственный
вектор; в направлении этого вектора данные обладают большей изменчивостью,
дисперсией.
Так называемые principal component scores - это перемножение, проекция данных
на тот или иной собственный вектор.
В данном случае, мы считаем, что основных компонент в данных всего две и
проецируемданные на них.
'''

pca = PCA().fit(spikes)

# get the PC scores and the eigenspectrum
pcscores = pca.transform(spikes)
explVar = pca.explained_variance_
explVar = 100*explVar/np.sum(explVar) # convert to % of total explained variance
coeffs = pca.components_

# show the scree plot (a.k.a. eigenspectrum)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(explVar, 'kp-', markerfacecolor='k', markersize=10)
ax[0].set_xlabel('Component number')
ax[0].set_ylabel('Percent variance explained')

ax[1].plot(np.cumsum(explVar), 'kp-', markerfacecolor='k', markersize=10)
ax[1].set_xlabel('Component number')
ax[1].set_ylabel('Cumulative percent variance explained')

# show the first 2 PC scores
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.suptitle("Data in the principal component space", y=0.92)
ax1.plot(pcscores[:, 0], pcscores[:, 1], 'k.', markersize=0.75)
ax1.set(xlabel='Principal component 1', ylabel='Principal component 2')
# %%
'''
Выделяются два четких кластера. При помощи метода k-means мы можем определить,
какие "дорожки" из данных относятся к какому кластеру.

Интерпретация существования двух кластеров:
в данных имеются два независимых источника сигналов.
'''
data = (np.vstack((pcscores[:, 0], pcscores[:, 1]))).T

# k-mean clustering is performed here
kmeans = KMeans(n_clusters=2).fit(data)
groupidx = kmeans.predict(data) # group (cluster) labels
cents = kmeans.cluster_centers_ # calculated positions of centroids

fig, ax = plt.subplots(figsize=(10, 6))
colors = 'rg'
for j in range(2):
    # Строим точки, отвечающие группе j
    ax.plot(data[groupidx == j, 0],
            data[groupidx == j, 1],
            'o', markerfacecolor=colors[j], markeredgecolor=colors[j],
            markersize=0.75, label=f"cluster {j+1}")
# and now plot the centroid locations
ax.plot(cents[:, 0], cents[:, 1], 'ko', label='Experimental cetroids')
ax.legend()
ax.set(xlabel='x', ylabel='y', title='Result of K-mean clustering')
# %%
'''
Усредним и выведем данные, относящиеся к тому или иному кластеру
'''

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True,)
fig.subplots_adjust(hspace=0.2)

ax[0].plot(np.mean(spikes, axis=0))
ax[0].set(ylabel='Amplitude', title='Average of all spikes')
ax[0].grid()

ax[1].plot(np.mean(spikes[groupidx==0], axis=0), 'r', label='cluster 1')
ax[1].plot(np.mean(spikes[groupidx==1], axis=0), 'g', label='cluster 2')
ax[1].set(xlabel='Time points', ylabel='Amplitude',
          title='Average of cluster-1 and cluster-2 spikes')
ax[1].grid()
ax[1].legend()
