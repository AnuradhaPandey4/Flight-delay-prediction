#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


df19=pd.read_csv('Jan_2019_ontime.csv')
df20=pd.read_csv('Jan_2020_ontime.csv')


# In[3]:


df19.head()


# In[4]:


df20.head()


# In[5]:


df19.drop('Unnamed: 21',axis=1,inplace=True)
df20.drop('Unnamed: 21',axis=1,inplace=True)


# In[6]:


df19['OP_CARRIER_AIRLINE_ID'].unique()


# In[7]:


df19['OP_CARRIER'].unique()


# In[8]:


frames = [df19, df20]

df = pd.concat(frames)

df.reset_index(drop=True, inplace=True)


# In[9]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df19 = df19.fillna(method ='pad')
df20 = df20.fillna(method ='pad')


# In[12]:


df19.isnull().sum()


# In[13]:


pl_1=df.groupby('DAY_OF_MONTH')['CANCELLED'].count()
fig = go.Figure()
fig.add_trace(go.Bar(x=pl_1.index,y=pl_1.values,name='Cancelled'))
fig.add_trace(go.Scatter(x=pl_1.index, y=pl_1.values, line=dict(color='red'), name='Cancel trend'))
fig.update_layout(
    title="Cancelled flights vs day of month",
    xaxis_title="Day of month",
    yaxis_title="Cancel count",
)
fig.show()


# In[14]:


pl_2=df.groupby('OP_CARRIER')['CANCELLED'].count()
fig = go.Figure()
fig.add_trace(go.Bar(x=pl_1.index,y=pl_1.values,name='Cancelled'))
fig.add_trace(go.Scatter(x=pl_1.index, y=pl_1.values, line=dict(color='red'), name='Cancel trend'))
fig.update_layout(
    title="Cancelled flights vs ID of flight",
    xaxis_title="Day of month",
    yaxis_title="Cancel count",
)
fig.show()


# In[15]:


df1 = df19[['OP_CARRIER','ORIGIN','DEST','DAY_OF_MONTH','DEP_TIME','ARR_TIME','DISTANCE','CANCELLED']]
df1.head()


# In[16]:


df1.dtypes


# In[17]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[18]:


pd.options.mode.chained_assignment = None
df1['OP_CARRIER']= label_encoder.fit_transform(df1['OP_CARRIER'])
df1['ORIGIN']= label_encoder.fit_transform(df1['ORIGIN'])
df1['DEST']= label_encoder.fit_transform(df1['DEST'])


# In[19]:


df1.dtypes


# In[20]:


X=df1.drop('CANCELLED',axis=1)
y=df1['CANCELLED']


# In[21]:


X.shape


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[24]:


from sklearn.tree import DecisionTreeClassifier
algo = DecisionTreeClassifier()
algo.fit(X_train, y_train)


# In[25]:


predict_test = algo.predict(X_test)


# In[26]:


accuracy_score(y_test,predict_test)


# In[27]:


from sklearn.naive_bayes import GaussianNB
GN_nb = GaussianNB()
model = GN_nb.fit(X_train, y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


accuracy_score(y_test,y_pred)


# In[33]:


from keras.models import Sequential
from keras.layers import Dense


# In[39]:


model = Sequential([
    Dense(32, activation='relu', input_shape=(7,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])


# In[40]:


model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[41]:


hist = model.fit(X_train, y_train,
          batch_size=32, epochs=100,
          validation_data=(X_test, y_test))


# In[42]:


plt.plot(hist.history['loss'], label='Training loss')
plt.plot(hist.history['val_loss'], color='red', label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




