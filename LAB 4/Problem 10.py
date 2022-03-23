# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:15:06 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')

#%%
df1.boxplot(column=['Class','V28'])
#%%
df1.boxplot(column=['Amount','V20'])
#%%
df1.boxplot(column=['V22','Time'])