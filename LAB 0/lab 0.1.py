# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:42:24 2022

@author: User
"""
import math

print("\nEnter a positive number:")
a=int(input())
print("\nEnter a positive number:")
b=int(input())
print("\nEnter a positive number:")
c=int(input())
if(a+b>c and a+c>b and b+c>a):
    print("\nValid Triangle.")
    s=(a+b+c)/2
    area=math.sqrt(s*(s-a)*(s-b)*(s-c))
    print("\nArea is ",area)
else:
    print("\nIntegers are not valid for triangle sides.")