# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:59:50 2022

@author: User
"""
marks=[]
crdts=[]
let_grd=[]
result=[]

print("\nEnter the mark of the course:")
a=int(input())
marks.append(a)

print("\nEnter the credit for that course:")
a_credit=int(input())
crdts.append(a_credit)

print("\nEnter the mark of the course:")
b=int(input())
marks.append(b)

print("\nEnter the credit for that course:")
b_credit=int(input())
crdts.append(b_credit)

print("\nEnter the mark of the course:")
c=int(input())
marks.append(c)

print("\nEnter the credit for that course:")
c_credit=int(input())
crdts.append(c_credit)

sum2=sum(crdts)

for i in range(3):
    if marks[i]>=95 and marks[i]<=100:
        let_grd.append('A')
        result.append(crdts[i]*4.00)
    elif marks[i]>=85 and marks[i]<=94:
        let_grd.append('B')
        result.append(crdts[i]*3.5)
    elif marks[i]>=70 and marks[i]<=84:
        let_grd.append('C')
        result.append(crdts[i]*3.0)
    elif marks[i]>=60 and marks[i]<=69:
        let_grd.append('D')
        result.append(crdts[i]*2.5)
    elif marks[i]>=0 and marks[i]<=59:
        let_grd.append('F')
        result.append(crdts[i]*0.0)
print('\nGrades:')
print("\nFirst Course:",let_grd[0])
print("Second Course:",let_grd[1])
print("Third Course:",let_grd[2])
print("\nTerm GPA:",(sum(result)/sum2))  