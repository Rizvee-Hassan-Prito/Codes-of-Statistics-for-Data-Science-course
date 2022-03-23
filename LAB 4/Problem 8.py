# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 04:16:10 2022

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns

#%%
df1=pd.read_csv('dataset_lab04.csv')
sns.distplot(df1['Amount'],color='g')
print('\nSkewness for "Amount" column =',df1['Amount'].skew())

#%%
sns.distplot(df1['V2'],color='r')
print('\nSkewness for "V2" column =',df1['V2'].skew())

#%%
sns.distplot(df1['V12'],color='b')
print('\nSkewness for "V12" column =',df1['V12'].skew())