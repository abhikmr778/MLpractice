#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('test_scores.csv')


# In[3]:


x = df['math']


# In[4]:


y = df['cs']


# In[5]:


import math


# In[21]:


def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 0
    n = len(x)
    learning_rate = 0.0001
    flag = True
    temp = 0
    while(flag):
        y_pred = m_curr*x + b_curr
        cost = (1/n)*sum(val**2 for val in (y-y_pred))
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd
        print(f'm: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {iterations}')
        if(iterations!=0):
            if(math.isclose(cost,temp,abs_tol = 1e-20)):
                flag = False
        temp = cost
        iterations += 1


# In[22]:


gradient_descent(x,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




