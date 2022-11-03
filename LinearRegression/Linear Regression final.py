#!/usr/bin/env python
# coding: utf-8

# In[94]:


from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import seaborn as sns
sns.set_style('darkgrid')


# In[95]:


df = pd.read_csv(r'C:\Users\hp\Downloads\APDS\Econometrics with R\RX_CACA5_Merged.csv')
data.columns


# In[96]:


df['mean_val'] = df[['deHorizontalVib', 'deVerticalVib', 'deAxialVib', 'nde_horizontalVib','nde_verticalVib', 'nde_axialVib']].mean(axis=1).copy()
df


# In[97]:


df1 =df['mean_val'].copy()
from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha = 0.05) 
adf_test.should_diff(df1)


# In[98]:


#df = df1[['time']].copy()
df['sno'] = pd.Series(np.arange(1,len(df)+1,1))
df


# In[99]:


reg = LinearRegression().fit(df[['sno']], df.mean_val)


# In[100]:


print(f"Intercept: {reg.intercept_}")
print(f"Slope: {reg.coef_}")
plt.xlabel("Count")
plt.ylabel("Vibration (Mean)")
#df['mean_val']=df['mean_val'].round(2)
plt.plot(df.sno, df.mean_val, color='blue')
#df3=df.append(df2)
df2 = pd.DataFrame(np.array([i for i in range(1,174)]), columns=['sno'])
df2['mean_val'] = np.array(reg.predict(df2))
#df2['mean_val']=df2['mean_val'].round(2)
#print(df2)
plt.plot(df2.sno, df2.mean_val, color='red')
plt.show()


# In[101]:


#If predicted value is greater than 2.8 then motor is rejected

print("******************************************************")
if (((df2.mean_val>2.8).any()) or reg.coef_[0] > 0):
    print("Motor test validation Failed, predicted value is greater than 2.8!!!")
else:
    print("Motor Validation Passed!!!")
print("******************************************************")


# In[ ]:




