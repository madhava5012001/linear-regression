#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:green" align="center">Machine Learning With Python: Linear Regression With One Variable</h2>

# <h3 style="color:purple">Sample problem of predicting home price in monroe, new jersey (USA)</h3>

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('homeprices.csv')


# In[17]:


df.info()


# In[4]:


df


# In[9]:


X=df.iloc[:,0:1]


# In[13]:


y=df[['price']]


# In[14]:


y


# In[10]:


X


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


plt.scatter(X,y,marker='*',color='red')


# In[62]:


X = df[['area']]


# In[63]:


X.shape


# In[64]:


df.area.shape


# In[65]:


y = df[['price']]


# In[19]:


from sklearn import linear_model
reg = linear_model.LinearRegression()


# In[21]:


reg.fit(X,y)


# In[22]:


reg.coef_


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.09, normalize=True)

ridgeReg.fit(X_train,y_train)
pred = ridgeReg.predict(X_train)
pred


# In[23]:


reg.fit(X,y)


# In[24]:


reg.coef_


# In[25]:


reg.intercept_


# In[120]:


reg.predict (X_train)


# In[77]:


X_vaildate = pd.read_csv('areas.csv')


# In[78]:


reg.predict(X_vaildate)

= mx+b
= [135.78767123](3100) + 180616.43835616

=  
# In[79]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[80]:


df = pd.read_csv('homeprices.csv')
df


# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[82]:


new_df = df.drop('price',axis=1)
new_df


# In[83]:


price = df.price
price


# In[84]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# **(1) Predict price of a home with area = 3300 sqr ft**

# In[85]:


reg.predict([[3300]])


# In[86]:


reg.coef_


# In[87]:


reg.intercept_


# **Y = m * X + b (m is coefficient and b is intercept)**

# In[88]:


3300*135.78767123 + 180616.43835616432


# **(1) Predict price of a home with area = 5000 sqr ft**

# In[89]:


y_pred = reg.predict([[3000]])


# In[90]:


y_pred


# In[91]:


y_actual = 565000


# In[92]:


import numpy as np
mse = np.sum((y_pred - y_actual)**2)
m=3
rmse = np.sqrt(mse/m)


# In[93]:


mse


# In[94]:


rmse


# In[95]:


# sum of square of residuals
ssr = np.sum((y_pred - y_actual)**2)

#  total sum of squares
sst = np.sum((y_actual - np.mean(y_actual))**2)

# R2 score
r2_score = 1 - (ssr/sst)


# In[96]:


ssr


# In[97]:


sst


# <h3 style="color:purple">Generate CSV file with list of home price predictions</h3>

# In[29]:


area_df = pd.read_csv("areas.csv")
area_df.head(3)


# In[30]:


p = reg.predict(area_df)
p


# In[31]:


area_df['prices']=p
area_df


# In[32]:


area_df.to_csv("prediction.csv")


# In[ ]:





# In[33]:


import pandas as pd


# In[34]:


df=pd.read_csv('predicted.csv')


# In[35]:


df


# In[36]:


df.columns


# In[37]:


df.isnull().sum()


# In[47]:


X=df[['Unnamed: 0']]


# In[48]:


y=df['0']


# In[49]:


from sklearn import linear_model
reg = linear_model.LinearRegression()


# In[50]:


reg.fit(X,y)


# In[51]:


reg.predict([[14]])


# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)


# In[53]:


from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)


# In[54]:


ridgeReg


# In[58]:


ridgeReg.fit(X_train,y_train)
pred = ridgeReg.predict(X_train)


# In[59]:


pred


# In[60]:


ridgeReg.predict([[14]])


# In[ ]:




