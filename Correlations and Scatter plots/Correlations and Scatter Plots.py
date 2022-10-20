# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:36:42 2022

@author: User
"""

import pandas as pd

datas=pd.read_csv("cereal.csv")

cols=['calories', 'protein', 'fat', 'sugars', 'carbo']

#%%
import matplotlib.pyplot as plt
import random

c=1
fig, ax = plt.subplots(figsize=(25, 20))
for i in range(len(cols)):
  for j in range(i+1, len(cols), 1):
    plt.subplot(5,5,c)
    plt.xlabel(cols[i])
    plt.ylabel(cols[j])
    plt.scatter(datas[cols[i]], datas[cols[j]])
    print("Correlation score between" ,cols[i],"and", cols[j],":", datas[cols[i]].corr(datas[cols[j]]), "\n")
    c+=1

#%%

import matplotlib.pyplot as plt
import random

colors=["blue", "green", "red", "cyan", "magenta", "orange", "black"]
markers=["X", "^", "o", "d", "*", "h"] 
c=1
fig, ax = plt.subplots(figsize=(25, 20))
for i in range(len(cols)):
  for j in range(i+1, len(cols), 1):
    plt.subplot(5,5,c)
    plt.xlabel(cols[i], color= "red")
    plt.ylabel(cols[j], color= "red")
    plt.scatter(datas[cols[i]], datas[cols[j]], color=colors[random.randint(0,5)], marker=markers[random.randint(0,5)])
    print("Correlation score between" ,cols[i],"and", cols[j],":", datas[cols[i]].corr(datas[cols[j]]), "\n")
    c+=1