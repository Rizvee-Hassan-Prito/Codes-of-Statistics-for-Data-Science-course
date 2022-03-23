# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:19:55 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')


#%%
df1.plot.scatter(x='V27', y='Amount')
#%%
df1.plot.scatter(x='Time', y='V22')
#%%
df1.plot.scatter(x='Amount', y='V20')