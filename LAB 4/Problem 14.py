# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:15:16 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')

#%%
df1.plot.scatter(x='Amount', y='V1')
#%%
df1.plot.scatter(x='Time', y='V28')
#%%
df1.plot.scatter(x='V19', y='Amount')