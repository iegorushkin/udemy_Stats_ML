# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:14:08 2021

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Number of samples
N = 10000

# Samples from normal distribution
normal = np.random.randn(N)

# Samples from uniform distribution
uniform = np.random.rand(N)

# Plot!
fig, ax = plt.subplots(1, 2, figsize=(8, 6))
sns.boxplot(data=normal, ax=ax[0])
sns.boxplot(data=uniform, ax=ax[1])
ax[0].set_title(f'Boxplot of {N}-sampled normal distribution', fontsize=12)
ax[1].set_title(f'Boxplot of {N}-sampled uniform distribution', fontsize=12)
