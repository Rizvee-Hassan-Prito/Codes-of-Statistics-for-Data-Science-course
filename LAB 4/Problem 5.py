# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 02:52:59 2022

@author: User
"""

"""
Compute the mean of any column using your own module and compare it with the mean value of
Pandas.
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
df1.hist(column=['Time'])
df1.hist(column=['Amount'])