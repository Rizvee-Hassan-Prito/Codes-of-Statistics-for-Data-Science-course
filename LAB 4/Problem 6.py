# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 01:09:26 2022

@author: User
"""

import pandas as pd

df1=pd.read_csv('dataset_lab04.csv')
c1=0
c2=0
for i in df1['Class']:
    if i==0:
        c1+=1
    else:
        c2+=1

print('Perceentage of non-fraudulent is:',(c1/(c1+c2))*100)

print('Perceentage of fraudulent is:',(c2/(c1+c2))*100)
