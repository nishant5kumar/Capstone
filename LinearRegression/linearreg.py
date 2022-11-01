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
df = pd.read_csv('C:\\Users\\nishant.x.kumar\\Desktop\\Project\\Capstone\\code\\RXCACA3_Merged_2.csv')
#print(df)
reg = LinearRegression().fit(df[['sno']], df.mean_val)
print(f"Intercept: {reg.intercept_}")
print(f"Slope: {reg.coef_}")
plt.xlabel("Count")
plt.ylabel("Vibration (Mean)")
#df['mean_val']=df['mean_val'].round(2)
plt.plot(df.sno, df.mean_val, color='blue')
#df3=df.append(df2)
df2 = pd.DataFrame(np.array([i for i in range(1,100)]), columns=['sno'])
df2['mean_val'] = np.array(reg.predict(df2))
#df2['mean_val']=df2['mean_val'].round(2)
#print(df2)
plt.plot(df2.sno, df2.mean_val, color='red')
plt.show()
#If predicted value is greater than 2.8 then motor is rejected

print("******************************************************")
if ((df2.mean_val>2.8).any()):
    print("Motor test validation Failed, predicted value is greater than 2.8!!!")
else:
    print("Motor Validation Passed!!!")
print("******************************************************")