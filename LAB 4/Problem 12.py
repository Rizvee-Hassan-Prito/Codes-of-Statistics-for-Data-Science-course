# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:24:07 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
#print(df1.corr())

lst1=[]
columns=list(df1.columns)
for i in columns:
    for j in columns:
        lst2=[]
        if(df1[i].corr(df1[j])<0):
            lst2.append(i)
            lst2.append(j)
            lst2.append(round(df1[i].corr(df1[j]),4))
            lst1.append(lst2)
            #print(f'Columns- {i} and {j}, Correlation value: {df1[i].corr(df1[j])}')
print('\nThese are the list of columns with negative correlation value:\n\n',lst1)