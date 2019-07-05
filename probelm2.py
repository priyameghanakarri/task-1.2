#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras import models
from keras import layers
from keras.layers import Dense


# In[2]:


# seed random number generator
seed(1)
# generate some integers
m= randint(-10,10,50)
#print(m)


# In[3]:


seed(2)
n=randint(-10,10,50)
#n


# In[5]:


#2*x:0,x*x:1,x*y:2,3*x+4*y:3
def a(num):
    for x in num:
        yield x
x1=a(m)
y1=a(n)
result=[]
function_used=[]
for k in range(0,4):
    for x,y in zip(x1,y1):
        if(k==0):
            res=2*x
            #result.append(res)
        elif(k==1):
            res=x*x
        elif(k==2):
            res=x*y
        else:
            res=3*x+4*y
        result.append(res)
        function_used.append(k)
    x1=a(m)
    y1=a(n)
print(len(result))
print(len(function_used))


# In[6]:


m=list(m)*4
n=list(n)*4
print(len(m))
dataset = pd.DataFrame({'x':m[:],'y':n[:],'result':result[:],'function_used':function_used[:]})
#print(dataset.head())


# In[7]:


dataset.to_csv(r'problem2work.csv')


# In[8]:


data=pd.read_csv('problem2work.csv',index_col=0)


# In[9]:


data.head()


# In[10]:


data=shuffle(data, random_state=0)


# In[11]:


data


# In[12]:


x=data.iloc[:, :-1]
y=data['function_used']


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)


# In[14]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[15]:


model = models.Sequential()

model.add(layers.Dense(256, activation='sigmoid', input_shape=(x_train.shape[1],)))

model.add(layers.Dense(128, activation='sigmoid'))

model.add(layers.Dense(64, activation='sigmoid'))

#model.add(layers.Dense(128, activation='sigmoid'))

#model.add(layers.Dense(64,activation='sigmoid'))

model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

#history=model.fit(x_train,y_train,epochs=20,batch_size=32)


# In[18]:


history=model.fit(x_train,y_train,epochs=200,batch_size=32)


# In[19]:


z=model.evaluate(x_test,y_test)


# In[20]:


z


# In[35]:


res=model.predict(x_test)

