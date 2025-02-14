# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:59:26 2022

@author: Igor
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from random import choice

## Setting variables that do not change in the loop.
n = 70  # number of points per cluster
A, B, C = [2, 5], [5, 2], [8, 8]  # x-y coordinates of cluster centroids
# Possible values of 'smearing' along the x and y axes (std units)
smearing = [0.5, 1, 2]
# epsilon parameter of DBSCAN
# epsilons = [0.5, 0.8, 1.2]
epsilons = 1
# for visualization
fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)
ax = ax.flatten()
colors = 'rgbmygrymb'
# %%

for i in range(3):
    ## Data generation
    a_smearing, b_smearing, c_smearing = ([choice(smearing), choice(smearing)],
                                          [choice(smearing), choice(smearing)],
                                          [choice(smearing), choice(smearing)])
    # now, simulate the data
    a = [A[0] + a_smearing[0]*np.random.default_rng().standard_normal(size=n),
         A[1] + a_smearing[1]*np.random.default_rng().standard_normal(size=n)]
    b = [B[0] + b_smearing[0]*np.random.default_rng().standard_normal(size=n),
         B[1] + b_smearing[1]*np.random.default_rng().standard_normal(size=n)]
    c = [C[0] + c_smearing[0]*np.random.default_rng().standard_normal(size=n),
         C[1] + c_smearing[1]*np.random.default_rng().standard_normal(size=n)]
    # concatenate all the data in 1 matrix
    data = (np.hstack((a, b, c))).T

    ## K-means clustering
    # k-mean clustering is performed here
    kmeans = KMeans(n_clusters=3).fit(data)
    groupidx_1 = kmeans.predict(data)  # group (cluster) labels
    cents_1 = kmeans.cluster_centers_  # calculated positions of centroids

    ## DBSCAN
    dbscan = DBSCAN(eps=epsilons, min_samples=6).fit(data)
    groupidx_2 = dbscan.labels_   # каждой точке ставится в соответствие лейбл
    # number of clusters
    nclust = max(groupidx_2) + 1  # +1 for indexing
    # compute cluster centers
    cents_2 = np.zeros((nclust, 2))
    for ci in range(nclust):
        # берем точки из data, индекс которых соответствует лейблу ci
        # усредняем координаты точек одной группы, получаем координаты её цетроида
        cents_2[ci, 0] = np.mean(data[groupidx_2 == ci, 0])
        cents_2[ci, 1] = np.mean(data[groupidx_2 == ci, 1])

    # Plot everything
    # Raw data
    ax[i].plot(data[:, 0], data[:, 1], '*')
    ax[i].plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ms',
               label='centroids', markersize=10)
    subplot_title = ('Raw data. \nCluster smearings:' +
                     f' a - {a_smearing}, b - {b_smearing}, c - {c_smearing}')
    ax[i].set_title(subplot_title, fontsize=11)

    # K-means clustering
    # draw lines from each data point to the centroids of each cluster
    for j in range(0, len(data)):
        ax[3+i].plot([data[j, 0], cents_1[groupidx_1[j], 0]],
                     [data[j, 1], cents_1[groupidx_1[j], 1]],
                     colors[groupidx_1[j]])
    # draw the raw data in different colors
    for j in range(3):
        # Строим точки, отвечающие группе j
        ax[3+i].plot(data[groupidx_1 == j, 0],
                     data[groupidx_1 == j, 1],
                     'o', markerfacecolor=colors[j], markersize=6)
    # plot the centroid locations
    ax[3+i].plot(cents_1[:, 0], cents_1[:, 1], 'ko', label='centroids',
                 markersize=10)
    ax[3+i].set_title('K-means clustering')

    # DBSCAN
    # draw lines from each data point to the centroids of each cluster
    for j in range(len(data)):
        if groupidx_2[j] == -1:
            ax[6+i].plot(data[j, 0], data[j, 1], 'k+', markersize=6) # это шум
        else:
            ax[6+i].plot([data[j, 0], cents_2[groupidx_2[j], 0]],
                         [data[j, 1], cents_2[groupidx_2[j], 1]],
                         colors[groupidx_2[j]])
    # draw the raw data in different colors
    for j in range(nclust):
        # Строим точки, отвечающие группе j
        ax[6+i].plot(data[groupidx_2 == j, 0],
                     data[groupidx_2 == j, 1],
                     'o', markerfacecolor=colors[j], markersize=6)
    # plot the centroid locations
    ax[6+i].plot(cents_2[:, 0], cents_2[:, 1], 'p', label='centroids',
                 markersize=10, markerfacecolor='white')
    ax[6+i].set_title(f'DBSCAN. Epsilon = {epsilons}')

'''
Mike:
I also have the feeling that k-means works better in this method
 of data simulation (although I wouldn't argue that k-means is better than
dbscan for all datasets).
'''
# %%
## Nonlinear clusters

# Generate the data - one circle inside another circle
N = 1000
th = np.linspace(0, 2*np.pi, N)
# create the two circles               аргументы задают форму выходного массива
data1 = np.array((np.cos(th), np.sin(th))) + np.random.randn(2, N) / 15
data2 = 0.3*np.array((np.cos(th), np.sin(th))) + np.random.randn(2, N) / 15
# put them together into one dataset
circdata = np.hstack((data1, data2)).T

# K-means clustering
kmeans = KMeans(n_clusters=2).fit(circdata)
groupidx_1 = kmeans.predict(circdata)  # group labels
cents_1 = kmeans.cluster_centers_  # calculated positions of centroids

# DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=8).fit(circdata)
groupidx_2 = dbscan.labels_   # group labels
# number of clusters
nclust = max(groupidx_2) + 1  # +1 for indexing
# compute cluster centers
cents_2 = np.zeros((nclust, 2))
for ci in range(nclust):
    cents_2[ci, 0] = np.mean(circdata[groupidx_2 == ci, 0])
    cents_2[ci, 1] = np.mean(circdata[groupidx_2 == ci, 1])

# Visualization
fig, ax = plt.subplots(3, 1, figsize=(5, 10), sharex=True, sharey=True)
ax = ax.flatten()
# Raw data
ax[0].plot(circdata[:, 0], circdata[:, 1], 'ko')
ax[0].axis('square')
ax[0].set_title('Raw data')
# K-means clustering
# draw the raw data in different colors
for j in range(2):
    # Строим точки, отвечающие группе j
    ax[1].plot(circdata[groupidx_1 == j, 0],
               circdata[groupidx_1 == j, 1],
               'o', markerfacecolor=colors[j], markersize=6)
# plot the centroid locations
ax[1].plot(cents_1[:, 0], cents_1[:, 1], 'ko', label='centroids',
           markersize=10)
ax[1].axis('square')
ax[1].set_title('K-means clustering')
# DBSCAN
# draw the raw data in different colors
for j in range(-1, nclust):
    # Строим точки, отвечающие группе j
    if j == -1:
        ax[2].plot(circdata[groupidx_2 == j, 0],
                  circdata[groupidx_2 == j, 1],
                   'k+', markersize=6)  # это шум
    else:
        ax[2].plot(circdata[groupidx_2 == j, 0],
                   circdata[groupidx_2 == j, 1],
                   'o', markerfacecolor=colors[j], markersize=6)
# plot the centroid locations
ax[2].plot(cents_2[:, 0], cents_2[:, 1], 'p', label='centroids',
           markersize=10, markerfacecolor='white')
ax[2].axis('square')
ax[2].set_title('DBSCAN. \nEpsilon = 0.25, min. samples = 8')
fig.tight_layout()

'''
Mike:
The results are going to be that K means is just gonna cut this thing
in half somewhere and put half of these dots into one cluster
and all dots into another cluster.
Exactly where it cuts in half is just going to depend on the random amount
of clustering in here.
'''
