#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv


# In[2]:


a = []


# In[3]:


with open('enjoysport.csv','r') as csvfile :
    for row in csv.reader(csvfile) :
        a.append(row)
    print(a)
print("\n The total number of training instances are : ",len(a))
num_attribute = len(a[0])-1


# In[4]:


print("\n Initial hypothesis is: ")
hypothesis = ['0']*num_attribute
print(hypothesis)
for i in range(0,len(a)):
    if a[i][num_attribute] == 'yes':
        for j in range(0,num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j] :
                hypothesis[j] = a[i][j]
            else :
                hypothesis[j] = '?'
            print("\n The hypothesis for training instance {} is \n" .format(i+1),hypothesis)
print("\n The maximally specific hypothesis for the training instance is")
print(hypothesis)

