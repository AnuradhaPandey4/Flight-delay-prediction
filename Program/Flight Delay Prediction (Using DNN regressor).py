import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

df19=pd.read_csv('Jan_2019_ontime.csv')
df20=pd.read_csv('Jan_2020_ontime.csv')

df19.head()

df20.head()

df19.drop('Unnamed: 21',axis=1,inplace=True)
df20.drop('Unnamed: 21',axis=1,inplace=True)

df19['OP_CARRIER_AIRLINE_ID'].unique()

df19['OP_CARRIER'].unique(


frames = [df19, df20]

df = pd.concat(frames)

df.reset_index(drop=True, inplace=True)

df.head()

df.isnull().sum()

df19 = df19.fillna(method ='pad')
df20 = df20.fillna(method ='pad')

df19.isnull().sum()

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

df1 = df19[['OP_CARRIER','ORIGIN','DEST','DAY_OF_MONTH','DEP_TIME','ARR_TIME','DISTANCE','CANCELLED']]
df1.head()

df1.dtypes

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

pd.options.mode.chained_assignment = None
df1['OP_CARRIER']= label_encoder.fit_transform(df1['OP_CARRIER'])
df1['ORIGIN']= label_encoder.fit_transform(df1['ORIGIN'])
df1['DEST']= label_encoder.fit_transform(df1['DEST'])

df1.dtypes

X=df1.drop('CANCELLED',axis=1)
y=df1['CANCELLED']

X.shape

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

input_func= tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train.astype(np.float32), 
                                                y= y_train.astype(np.float32), 
                                                batch_size=32, 
                                                num_epochs=1000, 
                                                shuffle=True)

eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_val.astype(np.float32),
                                                      y=y_val.astype(np.float32), 
                                                      batch_size=32, 
                                                      num_epochs=1, 
                                                      shuffle=False)
    

opti = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

estimator = tf.estimator.DNNRegressor(hidden_units=[128,256], feature_columns=feature_cols, optimizer=opti)

estimator.train(input_fn=input_func,steps=200)

result_eval = estimator.evaluate(input_fn=eval_input_func)

result_eval

