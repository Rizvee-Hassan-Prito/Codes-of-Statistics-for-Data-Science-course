# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:19:39 2022

@author: User
"""

"""
1. Find all of the numbers from 1–1000 that are divisible by 8.
"""

nums = [i for i in range(1,1001)]

filtered_nums= filter(lambda z:(z%8==0), nums)

print("\nNumbers that are disivible by 8 from 1 to 1000:\n")
for i in filtered_nums:
    print(i,end=" ")

"""
2. Find all of the numbers from 1–1000 that have a 6 in them.
"""
nums = [i for i in range(1,1001)]
filtered_nums= filter(lambda n:('6'in str(n)), nums)
print("\n\nNumbers that have '6' from 1 to 1000:\n")
for i in filtered_nums:
    print(i,end=" ")
    
"""
3. Count the number of spaces in a string (use string above)
"""

str1= "Practice Problems to Drill List Comprehension in Your Head."
print("\nString:",str1)
print("\nNumber of spaces in a string:",str1.count(" "))


"""
4. Remove all of the vowels in a string (use string above)
"""

vowels = ['a', 'e', 'i', 'o', 'u']
lt="Practice Problems to Drill List Comprehension in Your Head."
filteredVowels = filter(lambda i: (i not in vowels) , lt)
print("\nString:",lt)
print("\nNew string:","".join(list(filteredVowels)))

"""
5. Find all of the words in a string that are less than 5 letters (use string above)
"""
def lessThan5(lt):    
    if(len(lt)<5):
        return lt
    else:
        return ""

lt="Practice Problems to Drill List. Comprehension in Your Head."
print("\nString:",lt)
lt=lt.replace('.','')
words=list(map(lessThan5,lt.split()))
print("\nWords in the string that are less than 5 letters:\n")

for i in words:
    if i!="":
        print(i)

"""
6. Use a dictionary comprehension to count the length of each word in a sentence (use string above)
"""

string = "Practice Problems to Drill List Comprehension in Your Head."
print("\nLength of each word:\n")
string=string.replace('.', '')
lst=string.split(" ")
dct={i:len(i) for i in lst}
for i,j in dct.items():
    print(i,":",j)

"""
7. Use a nested list comprehension to find all of the numbers from 1–1000 that are divisible by any
single digit besides 1 (2–9)
"""
single_dgt=[2,3,4,5,6,7,8,9]
print("\nNumbers from 1–1000 divisible by any single digit:\n")
nums=[[j for j in range(1,1001) if (j%i==0)] for i in single_dgt]
nums2=[y for x in nums for y in x ]
nums2=list(set(nums2))
for i in nums2:
    print(i)

"""
8. For all the numbers 1–1000, use a nested list/dictionary comprehension to find the highest single
digit any of the numbers is divisible by
"""
div_hghst_num={k:v for k in range(1,1001) for v in range(1,10) if(k%v==0)}
print("\nThe highest single digit for the number of 1–1000 is:\n")
for i,j in div_hghst_num.items():
    print(i,':',j)
    
"""
With nested list Comprehension:
"""
div_hghst_num=[[i,j] for i in range(1,1001) for j in range(1,10) if(i%j==0)]
div_hghst_num2=[div_hghst_num[i] for i in range(0,len(div_hghst_num)-1) if(div_hghst_num[i][0] != div_hghst_num[i+1][0])]
div_hghst_num2.append(div_hghst_num[len(div_hghst_num)-1])
print("\nThe highest single digit for the number of 1–1000 is:\n")
for i in div_hghst_num2:
    print(i[0],':',i[1])