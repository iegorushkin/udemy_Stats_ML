# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:46:53 2021

@author: Igor
"""
import pandas as pd
import numpy as np

import seaborn as sns
# sns.axes_style
# sns.set_style

import matplotlib.pyplot as plt
# plt.style.available
# plt.style.use()


# number of samples for both distributions
n = 1000

rng = np.random.default_rng()

# Dataset 1
std = 1.5
mu = 5
dataset1 = rng.normal(loc=mu, scale=std, size=n)

# Dataset 2
dataset2 = rng.lognormal(size=n)

# Visualize these distributions
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
fig.subplots_adjust(hspace=0.3)

ax[0].hist(dataset1, bins='fd', histtype='bar', linewidth=1,
           edgecolor='black')
ax[0].set(xlim=[0, 16], xlabel='values', ylabel='counts',
          title='Normal distribution')

ax[1].hist(dataset2, bins='fd', color='r', linewidth=1, edgecolor='black')
ax[1].set(xlabel='values', ylabel='counts',
          title='Lognormal distribution')
#%%
# Violin plot

# Normal distr
temp1 = np.zeros((n, 2))
temp1[:, 1] = dataset1
# Lognormal distr
temp2 = np.ones((n, 2))
temp2[:, 1] = dataset2
# Dummy column
# Seaborn needs it to plot a proper asymmetric violin
temp3 = np.full((2*n, 1), fill_value=2)

# Concat all
data = np.hstack((np.vstack((temp1, temp2)), temp3))
del temp1, temp2, temp3

# Create DataFrame
data_df = pd.DataFrame(
    data=data, columns=['Distribution', 'Value', 'Temp']
)
data_df.replace(
    inplace=True, to_replace=[0, 1, 2],
    value=['Normal distribution', 'Lognormal distribution', 'Dummy']
)
print(data_df.head())

# Plot
fig, ax = plt.subplots()
sns.violinplot(data=data_df, x='Temp', y='Value',
               hue='Distribution', inner="quartile",
               split=True, palette=["lightblue", "lightpink"],
               orient='v', ax=ax)
ax.tick_params(axis='x', which='both', bottom=False,
               labelbottom=False)
ax.set(ylim=[-1.5, 15], xlabel='Counts')
ax.set_title('Asymmetric violin plot', fontsize=14)
