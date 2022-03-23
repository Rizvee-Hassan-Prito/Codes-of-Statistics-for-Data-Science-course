# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:06:29 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')

#%%
df1.boxplot(column=['Class','V17'])
#%%
df1.boxplot(column=['Amount','V5'])
#%%
df1.boxplot(column=['V24','Time'])