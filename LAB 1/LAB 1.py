# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:30:18 2022

@author: User
"""

"""
1. Given two integer numbers, write a Python program to return their product. If the product is greater
than 1000, then return their sum. Read inputs from the user.
"""
a=int(input("Give an integer number: "))
b=int(input("Give an integer number: "))

print("\nProduct=",a*b)
if((a*b)>1000):
    print("\nSum=",a+b)
    
"""
2. Write a Python program to find the area and perimeter of a circle. Read inputs from the user.
"""
import math
r=int(input("Enter the value of radius: "))
area=math.pi*(r**2)
perimeter=2*math.pi*r
print(f'\nArea = {area} and Perimeter = {perimeter}')

"""
3. Write a Python program to calculate the compound interest based on the given formula. Read inputs
from the user.
A = P * (1 + R/100)^T 
where P is the principle amount, R is the interest rate and T is time (in years).
Define a function named as compound_interest_<your-student-id> in your program.
"""

def compound_interest_2019_3_60_041(P,R,T):
    A= P*(1+(R/100))**T
    return A    

P=int(input("Enter principal Amount: "))
R=int(input("Enter Interest Rate: "))
T=int(input("Enter Time (in years): "))
print(f'\nCompound Interest {compound_interest_2019_3_60_041(P, R, T)} Taka')

"""
4. Given a positive integer N (read from the user), write a Python program to calculate the value of the following series.
                1^2 + 2^2 + 3^2 + 4^2 ..... + N^2
"""
n=int(input("Enter a positive integer:"))
sum1=0
for i in range(1,n+1):
    sum1+=i*i
print("\nSum is",sum1)

"""
5. Given a positive integer N (read from the user), write a Python program to check if the number is
prime or not. Define a function named as prime_find_<your-student-id> in your program.
"""

def prime_find_2019_3_60_041(n):
    if(n==0 or n==1):
        return False
    for i in range(2,n):
        if(n%i==0):
            return False
    return True

n=int(input("Enter a positive integer: "))
check=prime_find_2019_3_60_041(n)
if(check==True):
    print("\nPrime.")
else:
    print("\nNot Prime.")
    
"""
6. Given a positive integer n (read from the user), write a Python program to find the n-th Fibonacci
number based on the following assumptions.
Fn = Fn-1 + Fn-2 where F0 = 0 and F1 = 1
"""
n=int(input("Enter a positive integer: "))

fib=[0,1]
sum_fib=0

for i in range(2,n+1):
    sum_fib = fib[len(fib)-1] + fib[len(fib)-2]
    fib.append(sum_fib)

if(n==1):sum_fib=1
print(f'\n{n}th Fibonacci is: {sum_fib}')

"""
7. Given a list of numbers (hardcoded in the program), write a Python program to calculate the sum of
the list. Do not use any built-in function.
"""
lst=[1,3,4,5]
sum1=0
for i in lst:
    sum1+=i
print(f'\nSum is {sum1}')

"""
8. Given a list of numbers (hardcoded in the program), write a Python program to calculate the sum of
the even-indexed elements in the list.
"""
lst=[0,1,2,3,4,5,6,7,8,9,10]
sum1=0
for i in range(0,len(lst),2):
    sum1+=i
print(f'\nSum is {sum1}')

"""
9. Given a list of numbers (hardcoded in the program), write a Python program to find the largest and
smallest element of the list. Define two functions largest_number_<your-student-id> and
smallest_number_<your-student-id> in your program. Do not use any built-in function.
"""
def largest_number_2019_3_60_041(lst):
    l=lst[0]
    for i in lst:
        if(l<i):
            l=i
    return l

def smallest_number_2019_3_60_041(lst):
    s=lst[0]
    for i in lst:
        if(s>i):
            s=i
    return s

lst=[5,2,6,9,11,3,7,1,4]

l=largest_number_2019_3_60_041(lst)
s=smallest_number_2019_3_60_041(lst)

print(f'\nLargest number in the list is {l}')
print(f'\nSmallest number in the list is {s}')

"""
10. Given a list of numbers (hardcoded in the program), write a Python program to find the second
largest element of the list.
"""
def second_largest_number_2019_3_60_041(lst):
    l=lst[0]
    s_l=l
    for i in lst:
        if(l<i):
            s_l=l
            l=i
    return s_l

lst=[5,2,6,9,11,3,7,1,4]

sl=second_largest_number_2019_3_60_041(lst)

print(f'\nSecond largest number in the list is {sl}')

"""
11. Given a string, display only those characters which are present at an even index number. Read inputs
from the user.
"""
n=input("Enter a string: ")

for i in range(0,len(n),2):
    print(f'{n[i]}')

"""
12. Given a string and an integer number n, remove characters from a string starting from zero up to n
and return a new string. N must be less than the length of the string. Read inputs from the user. Do
not use any built-in function.
"""
s=input("Enter a string: ")
n=int(input("Enter a positive integer(must be less than the length of the string): "))
st=""
len_s=0
for i in s:
    len_s+=1
for i in range(n+1,len_s):
    st+=s[i]
print(f'String is -> {st}')

"""
13. Given a string, find the count of the substring “CSE303” appeared in the given string. Do not use any
built-in function.
"""

s=input("Enter a string: ")
s2="CSE303"
len_s=0
for i in s:
    len_s+=1
count=0
for i in range(0,len_s-5):
    st=""
    for j in range(i,i+6):
        st+=s[j]
    if(st==s2):
        count+=1
print(f'\nNumber of "CSE303" appeared: {count}')

"""
14. Given a string, write a python program to check if it is palindrome or not. Define a function named
palindrome_checker_<your-student-id> in your program.
"""

def palindrome_checker_2019_3_60_041(st):
    st1=list(st)
    st1.reverse()
    st1="".join(st1)
    if(st==st1):
        return True
    return False

s=input("Enter a string:")
z=palindrome_checker_2019_3_60_041(s)

if(z==True):
    print("\nPalindrome.")
else:
    print("\nNot Palindrome.")

"""
15. Given a two list of numbers (hardcoded in the program), create a new list such that new list should
contain only odd numbers from the first list and even numbers from the second list.
"""

lst1=[1,4,3,6,7,5,12]
lst2=[11,14,13,16,17,15,18]
lst3=[]

for i in lst1:
    if(i%2!=0):
        lst3.append(i)
    
for i in lst2:
   if(i%2==0):
     lst3.append(i)
print(f'New list: {lst3}')