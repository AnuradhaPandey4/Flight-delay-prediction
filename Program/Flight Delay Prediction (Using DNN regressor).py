#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf


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


# In[38]:


feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[41]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


# In[42]:


input_func= tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train.astype(np.float32), 
                                                y= y_train.astype(np.float32), 
                                                batch_size=32, 
                                                num_epochs=1000, 
                                                shuffle=True)


# In[43]:


eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_val.astype(np.float32),
                                                      y=y_val.astype(np.float32), 
                                                      batch_size=32, 
                                                      num_epochs=1, 
                                                      shuffle=False)


# In[44]:


opti = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')


# In[45]:


estimator = tf.estimator.DNNRegressor(hidden_units=[128,256], feature_columns=feature_cols, optimizer=opti)


# In[46]:


estimator.train(input_fn=input_func,steps=200)


# In[47]:


result_eval = estimator.evaluate(input_fn=eval_input_func)


# In[48]:


result_eval

