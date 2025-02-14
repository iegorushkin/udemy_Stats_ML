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

# Create numpy Random Generator
rng = np.random.default_rng()

# Samples from normal distribution
normal = rng.standard_normal(N)

# Samples from uniform distribution
uniform = rng.uniform(size=N)

# Plot!
fig, ax = plt.subplots(2, 1, figsize=(6, 10))
sns.boxplot(data=normal, ax=ax[0])
sns.boxplot(data=uniform, ax=ax[1])
ax[0].set_title(f'Boxplot of {N}-sampled normal distribution', fontsize=12)
ax[1].set_title(f'Boxplot of {N}-sampled uniform distribution', fontsize=12);
