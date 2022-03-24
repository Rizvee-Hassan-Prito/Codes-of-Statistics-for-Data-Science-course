# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:27:08 2022

@author: User
"""
import pandas as pd
import numpy as np
from numpy import random
import scipy.stats
import seaborn as sns

df=pd.read_csv('student_age.csv')

df=list(df)
df=[float(y) for y in df]
df2=pd.DataFrame(df)
means_list=[]

for i in range(0,1000):
    x=random.choice(df,size=(20))
    x2=random.choice(x,size=(100))
    means_list.append(x2.mean())


means=pd.DataFrame(means_list)
mean=means.mean()

std=list(means.std())
print('\nStandard Deviation:',std[0])

std_error=list(means.sem())
print('\nStandard Error:',std_error[0])

dof=len(means)-1
CL=0.95

t_val=scipy.stats.t.ppf(q=1-CL/2,df=dof)

CI=[float(mean-(std[0]*t_val/np.sqrt(len(means)))) , float(mean+(std[0]*t_val/np.sqrt(len(means))))]

print('\nConfidence Interval:',CI)

#sns.distplot(means,color='g')


