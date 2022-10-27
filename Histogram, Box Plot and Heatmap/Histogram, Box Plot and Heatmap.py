# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:43:05 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("Boston.csv")

print(df)

#%%

df=df.drop("Unnamed: 0",axis=1)
print(df)

#%%
df.describe()

#%%

cols=list(df.columns)

for i in cols:
  plt.subplots(figsize = (10,5))
  plt.hist(df[i])
  plt.xlabel(i, fontsize=18)
  
#%%

for i in cols:
  plt.subplots(figsize = (10,5))
  plt.boxplot(df[i])
  plt.xlabel(i,fontsize=18)
  
  
#%%

def upbd_lowbd(lst):
    
    q1 = np.quantile(lst, 0.25)
     
    q3 = np.quantile(lst, 0.75)

    iqr = q3-q1
     
    # finding upper and lower whiskers
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    l =[upper_bound, lower_bound]
    return l

#%%

for i in cols:
  range=upbd_lowbd(list(df[i].values))
  values=list(df[i].values)
  c=0
  for j in values:
    if (range[0]<j or range[1]>j):
      c+=1
  print("Number of outliers in", i, 'is :', c)
  
#%%

import seaborn as sns

plt.figure(figsize = (16,10))

ax=sns.heatmap(df.corr(),annot=True,  linewidth=.3, fmt=".2f")

#ax.set(xlabel="", ylabel="")
#ax.xaxis.tick_top()