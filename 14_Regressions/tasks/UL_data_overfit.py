# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 21:47:59 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# generate 2 datasets - same n and trend, different noise
n = 10
trend = np.arange(10)
data1 = trend + np.random.default_rng().normal(loc=0, scale=1.2, size=n)
data2 = trend + np.random.default_rng().normal(loc=0, scale=1.5, size=n)

# plot
fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
fig.suptitle('The problem with overfitting', y=0.95, fontsize=16)

ax[0, 0].plot(data1, 's', markerfacecolor='w', markersize=8)
ax[0, 0].set_title('Dataset 1', fontsize=12)

ax[1, 0].plot(data2, 's', markerfacecolor='w', markersize=8)
ax[1, 0].set_title('Dataset 2', fontsize=12)
# %%
# fit data1 with 2 order and 10 order polynomial regressions

betas_under = np.polyfit(trend, data1, 1)  # highest power terms first!
model1 = np.polyval(betas_under, trend)
betas_over = np.polyfit(trend, data1, 9)
model2 = np.polyval(betas_over, trend)

# plot
ax[0, 1].plot(data1, 'sb', markerfacecolor='w', markersize=8)
ax[0, 1].plot(model1, 'mo', markerfacecolor='w', markersize=8)
ax[0, 1].set_title('2nd order polynomial fit', fontsize=12)
ax[0, 2].plot(data1, 'sb', markerfacecolor='w', markersize=8)
ax[0, 2].plot(model2, 'm*', markerfacecolor='w', markersize=8)
ax[0, 2].set_title('10th order polynomial fit', fontsize=12)

# does the model for dataset 1 apply to dataset 2?
ax[1, 1].plot(data2, 'sb', markerfacecolor='w', markersize=8)
ax[1, 1].plot(model1, 'mo', markerfacecolor='w', markersize=8)
ax[1, 1].set_title('Dataset 2 vs. model 1', fontsize=12)
ax[1, 2].plot(data2, 'sb', markerfacecolor='w', markersize=8)
ax[1, 2].plot(model2, 'm*', markerfacecolor='w', markersize=8)
ax[1, 2].set_title('Dataset 2 vs. model 2', fontsize=12)
# %%
# evaluate models fit with R squared

# loop-based calculations
data = np.array([data1, data2])
models = np.array([model1, model2])
SSt_data = np.zeros(2)

R = np.ones((2, 2))
for i in range(R.shape[0]):
    SSt_data[i] = np.sum((data[i] - np.mean(data[i]))**2)
    for j in range(R.shape[1]):
        R[i, j] -= np.sum((data[i] - models[j])**2)/SSt_data[i]
print(R)

# # Sum of squares

# # Total sum of squares for datasets 1 and 2
# SSt_data1 = np.sum((data1 - np.mean(data1))**2)
# SSt_data2 = np.sum((data2 - np.mean(data2))**2)

# # model1 vs data1
# SSe_data1_mod1 = np.sum((data1 - model1)**2)
# # model2 vs data1
# SSe_data1_mod2 = np.sum((data1 - model2)**2)
# # model1 vs data2
# SSe_data2_mod1 = np.sum((data2 - model1)**2)
# # model2 vs data2
# SSe_data2_mod2 = np.sum((data2 - model2)**2)

# R = np.ones((2, 2))

# R[0, 0] -= SSe_data1_mod1/SSt_data1
# R[0, 1] -= SSe_data1_mod2/SSt_data1

# R[1, 0] -= SSe_data2_mod1/SSt_data2
# R[1, 1] -= SSe_data2_mod2/SSt_data2

# print(R)
