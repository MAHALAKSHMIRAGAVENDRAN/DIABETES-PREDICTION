#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("diabetes.csv")
df


# In[5]:


#checking any null values and missing values
df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.shape
df.describe()


# In[8]:


# checking the duplicates value
df.duplicated()


# In[9]:


correlation_matrix=df.corr()
correlation_matrix


# In[10]:


sns.heatmap(correlation_matrix,annot=True,cmap='RdPu')
plt.title('correlation matrix')


# In[11]:


unique_value=df['Outcome'].unique()
print(unique_value)


# In[12]:


#to get the value count
df['Outcome'].value_counts()


# In[13]:


X=df.drop(columns='Outcome',axis=1)
y=df['Outcome']
print(X)
print(y)


# In[15]:


X.shape
y.shape


# In[18]:


from sklearn.preprocessing import StandardScaler

# Assuming you have your data X defined somewhere above

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on your data
scaler.fit(X)

# Now you can transform the data
standard_data = scaler.transform(X)

# Now you can use the standardized data
print(standard_data)


# In[21]:


# splitting training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(standard_data, y, test_size=0.2, random_state=42)


# In[22]:


print(X.shape,X_train.shape,y_train.shape)


# In[23]:


from sklearn import svm
model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)


# In[26]:


#making prediction
prediction=model.predict(X_test)
prediction
#model evaluation
from sklearn.metrics import accuracy_score
print("train score and test score of svm",model.score(X_train,y_train)*100)
print("test score of svm",model.score(X_test,y_test)*100)
print("Accuracy score of svm",accuracy_score(y_test,prediction)*100)


# In[ ]:




