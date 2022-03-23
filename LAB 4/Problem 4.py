# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 02:50:11 2022

@author: User
"""

"""
Compute the mean of any column using your own module and compare it with the mean value of
Pandas.
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')

def mean(lst):
    r=sum(lst)/len(lst)
    return r
lt=df1[['Amount']].values.tolist()
lt=[y for x in lt for y in x]
print(f'\nMean from the own module: {mean(lt)}')
print('\nMean from Pandas:\n')
print(df1[['Amount']].mean())
