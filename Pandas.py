# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:00:07 2022

@author: User
"""

import pandas as pd
import numpy as np

df=pd.read_csv('melb_data.csv')

"""
df_price=df[['Address','Price','Distance','Car','YearBuilt']]
print(df_price.head())
"""
df1=df[['Price', 'Rooms', 'Distance','Landsize']]
#print(df1.describe())
#df1['Price'].plot(kind='hist',xticks=[0,0.5,1.0,1.5,2.0,2.5,3.0])
#df1['Price'].plot(kind='box')
print(df1.describe())

"""
dict1 = {'id':[1,2,3],'name':['alice','bob','charlie'],'age':[20, 25, 32]}
df1 = pd.DataFrame(dict1)
print(df1)
print(df1.count())
"""
