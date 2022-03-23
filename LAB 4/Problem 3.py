# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 02:22:28 2022

@author: User
"""

"""
There are 31 columns in the dataset. Compute some statistical measures like mean, median, standard
deviation, variance using Pandas Function.
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
print('\nMean:\n')
print(df1.mean())
print('\nMedian:\n')
print(df1.median())
print('\nStandard Deviation:\n')
print(df1.std())
print('\nVariance:\n')
print(df1.var())