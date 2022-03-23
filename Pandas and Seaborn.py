# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:05:50 2022

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns
"""
"""

"""
obj = pd.Series([4, 7, -5, 3])
print(obj**2)
"""
"""
sdata = {3: 35000, 2: 71000, 7: 16000, '4': 5000}
obj=pd.Series(sdata)
print(obj)
"""
"""
list1 = [['Alice',23,3.5,10],['Bob',24,3.4,6],['Charlie',22,3.9,8]]
df = pd.DataFrame(list1)
df.columns = ['name','age','cgpa','hoursStudied']
print(df.head())
"""

df1=pd.read_csv('dataset_lab04.csv')

#df1.drop(['V1'], axis=1, inplace=True)
#print(df1)

#print(df1.count())

#columns=[df1.iloc[0][i] for i in range(0,31)]
#print(columns)

#print(df1[[0,8,12]])
#print(df1[['Time','Amount']]) # if header!=None
#print(df1.iloc[[5,13]])
#print(df1[3:9])

#print(df1.iloc[0:6,[0,1,2,3]])
#print(df1.iloc[[0,6],[0,1,2,3]])
#print(df1.iloc[2:5,3:7])

#print(df1.iloc[[2,3,4]].describe())
#print(df1[3:9].describe())

#print(df1[1:4].sum())
#print(df1.median())
#print(df1.skew())

#df1['V1'].value_counts().plot(kind = 'bar')
"""
df2={'ALICE':[1,34,24],'Bob':[3,37,11]}
df2=pd.DataFrame(df2)
df2.columns=['Charlie','Dina']
print(df2)
"""
#print(df1[['Time']].describe())
#print(df1.corr())
#df1.boxplot(column=['V1','V2','V3'])
#df1['V1'].plot(kind='box')
#sns.boxplot(y=df1["V3"])
#print(df1.skew())
#df1['Amount'].plot(kind='hist')
#df1.hist(column=['V1','V2','V3','Time','Amount'])

#df1['V1'].plot(kind='box')
#df1['Amount'].plot(kind='hist')
#df1['Amount'].value_counts().plot(kind='bar')
#df1.plot.scatter(x='Time', y='Amount',figsize=(100,10))
#sns.distplot(df1['Time'],color='g')
#sns.countplot(df1['Time'])
#sns.heatmap(df1.corr(),annot=True)
#sns.jointplot(x='Time',y='Amount', data=df1, kind='hex')
"""
def mean(lst):
    r=sum(lst)/len(lst)
    return r
lt=df1[['Amount']].values.tolist()
lt=[y for x in lt for y in x]
print(f'Mean from module: {mean(lt)}')
print(df1[['Amount']].mean())
"""
#print(df1[['Amount']].var())
