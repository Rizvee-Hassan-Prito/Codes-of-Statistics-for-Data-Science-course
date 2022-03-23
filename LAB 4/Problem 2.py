# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 02:19:03 2022

@author: User
"""

"""
Describe (numerical summary) the time and amount column.
"""
import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
print('\nNumerical summary of Time attribute:')
print(df1[['Time']].describe())

print('\nNumerical summary of Amount attribute:')
print(df1[['Amount']].describe())