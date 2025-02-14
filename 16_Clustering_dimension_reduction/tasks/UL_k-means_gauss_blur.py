# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 22:19:10 2022

@author: Igor
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# %%
# generate and visualize the data

n = 200 # number of points per cluster

# x-y coordinates of cluster centroids
A = [15, 15]

# cluster 'smearing' along the x and y axes parameter (std units)
a_smearing = [1, 5]

# data
a = [A[0] + a_smearing[0]*np.random.default_rng().standard_normal(size=n),
     A[1] + a_smearing[1]*np.random.default_rng().standard_normal(size=n)]
# concatenate the data in 1 matrix
data = (np.array(a)).T
print(data.shape)

# visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data[:, 0], data[:, 1], '*')
ax.plot([A[0]], [A[1]], 'ms', label='True centroids')
ax.legend()
ax.set(xlabel='x', ylabel='y', title='Raw data');
# %%
# let's apply k-means algorithm with a different number of k-clusters.

fig, ax = plt.subplots(2, 3, figsize=(12, 6))
ax = ax.flatten()
colors = 'rgbmygrymb'

# apply the k-means algorithm using a different number of clusters
for k in range(6):
    # k-mean clustering is performed here
    kmeans = KMeans(n_clusters=k+1).fit(data)
    groupidx = kmeans.predict(data)  # group (cluster) labels
    cents = kmeans.cluster_centers_  # calculated positions of centroids

    # draw lines from each data point to the centroids of each cluster
    for i in range(0, len(data)):
        ax[k].plot([data[i, 0], cents[groupidx[i], 0]],
                   [data[i, 1], cents[groupidx[i], 1]],
                   colors[groupidx[i]])

    # and now plot the centroid locations
    ax[k].plot(cents[:, 0], cents[:, 1], 'ko')
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    ax[k].set_title('%g clusters'%(k+1))
'''
Вывод: в этом наборе данных, как и в большинстве наборов данных в принципе,
практически невозможно определить "правильное" число кластеров.
'''
