# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:52:44 2022

@author: User
"""


# 1. Using Statsmodel OLS Method


import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

print('\nEnter file location/path:')
file_name=input()
data = pd.read_csv(file_name)

print('\nEnter columns:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break

data.columns=columns
#data.columns = ['Country','Newspapers','Radios','TVsets','literacyRate']

print("\nGiven dataset:")
print(data)


# Preparing X (independent variables) and Y (dependent variables)

print('\nEnter independent variable names:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break
#X = data[['Newspapers','Radios','TVsets']]
X = data[columns]
print('\nEnter dependent variable names:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break
#Y = data[['literacyRate']]
Y = data[columns]

print("\nIndependent Variables:\n")
print(X)
print("\nDependent Variable:\n")
print(Y)

# converting X and Y to numpy array

X = np.asarray(X, dtype = float)
Y = np.asarray(Y, dtype = float)


# adding another column representing bias x0 (constant value)

X = sm.add_constant(X)

#print(X)
#print(Y)


model = sm.OLS(Y, X) 
results = model.fit()

print(results.summary())

