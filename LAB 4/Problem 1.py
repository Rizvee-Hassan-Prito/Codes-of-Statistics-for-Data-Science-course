# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 02:09:51 2022

@author: User
"""

"""
How many rows and columns this dataframe has?
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
a=df1.shape
print(f'This dataframe has {a[0]} rows and {a[1]} columns')
