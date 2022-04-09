#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r'C:\Users\ralfy\OneDrive\Desktop\CSV Files\BAJFINANCE.csv')
df.head()


# In[3]:


df.set_index('Date',inplace=True)


# In[4]:


df['VWAP'].plot()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df.dropna(inplace=True)


# In[8]:


df.isna().sum()


# In[9]:


df.shape


# In[10]:


data=df.copy()


# In[11]:


data


# In[12]:


data.dtypes


# In[13]:


data.columns


# In[14]:


lag_features=['High','Low','Volume','Turnover','Trades']
window1=3
window2=7


# In[15]:


for feature in lag_features:
    data[feature+'rolling_mean_3']=data[feature].rolling(window=window1).mean()
    data[feature+'rolling_mean_7']=data[feature].rolling(window=window2).mean()


# In[16]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[17]:


data.head()


# In[18]:


data.columns


# In[19]:


data.shape


# In[20]:


data.isna().sum()


# In[21]:


data.dropna(inplace=True)


# In[22]:


data.isna().sum()


# In[23]:


data.columns


# In[24]:


ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[25]:


training_data=data[0:1800]
test_data=data[1800:]


# In[26]:


training_data


# In[27]:


get_ipython().system('pip install pmdarima')


# In[28]:


from pmdarima import auto_arima


# In[30]:


import warnings
warnings.filterwarnings('ignore')


# In[31]:


model=auto_arima(y=training_data['VWAP'],exogenous=training_data[ind_features],trace=True)


# In[32]:


model.fit(training_data['VWAP'],training_data[ind_features])


# In[34]:


forecast=model.predict(n_periods=len(test_data), exogenous=test_data[ind_features])


# In[35]:


test_data['Forcast_ARIMA']=forecast


# In[38]:


test_data[['VWAP','Forcast_ARIMA']].plot(figsize=(14,7))


# In[39]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[40]:


np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forcast_ARIMA']))


# In[41]:


mean_absolute_error(test_data['VWAP'],test_data['Forcast_ARIMA'])


# In[ ]:




