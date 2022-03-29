# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:45:22 2022

@author: User
"""

import numpy as np

l2=[]

print('\nEnter Rows number:')
r=int(input())
print('\nEnter Columns number:')
c=int(input())

print('\nEnter the elements of matrix row-wise:\n')
for i in range(0,r):
    l=[]
    for j in range(0,c):
        l.append(int(input()))
    l2.append(l)


M=np.array(l2)
cofac=np.zeros((3,3))

for i in range(0,3):
  for j in range(0,3):
    cofac[i][j]=(M[(i+1)%3][(j+1)%3]*M[(i+2)%3][(j+2)%3])-(M[(i+1)%3][(j+2)%3]*M[(i+2)%3][(j+1)%3])

print("\nCofactor Matrix:\n",cofac)
dtm=int()
for i in range(0,3):
  for j in range(0,3):
    if(i==0):
      dtm+=M[i][j]*((M[(i+1)%3][(j+1)%3]*M[(i+2)%3][(j+2)%3])-(M[(i+1)%3][(j+2)%3]*M[(i+2)%3][(j+1)%3]))

print("\nDeterminant:",dtm)

T_M=np.transpose(cofac)
print("\nTranspose Matrix:",T_M)

inverse=T_M*(1/dtm)
print("\nInverse Matrix:\n",inverse)