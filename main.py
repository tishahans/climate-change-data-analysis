import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
%matplotlib inline

df=pd.read_csv('/content/GlobalLandTemperaturesByState.csv')

df.head()

df.dtypes

df.shape

df.isnull().sum()

df = df.dropna(how='any' ,axis=0)

df.shape

df.rename(columns={'dt':'Date','AverageTemperature':'avg_temp','AverageTemperatureUncertainty':'confidence_interval_temp'},inplace=True)
df.head()

df.describe()


df ['Date'] = pd. to_datetime(df[ 'Date'])
df.set_index('Date',
inplace = True)
df.index

df.describe()

df['Year']=df.index.year
df.head()

df.describe()

latest_df = df. loc['1980': '2013']
latest_df.head()

latest_df[['Country', 'avg_temp']].groupby(['Country']).mean().sort_values ('avg_temp' )

plt. figure(figsize=(9, 4))
sns.lineplot(x ="Year", y = "avg_temp", data=latest_df)
plt.show()

resample_df = latest_df[['avg_temp']].resample('A').mean()

resample_df.head()

resample_df.plot(title='Temperature Changes from 1980-2013' ,figsize=(8,5))
plt.ylabel( 'Temperature' ,fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()

from statsmodels.tsa.stattools import adfuller
print( 'Dickey Fuller Test Results: ')
test_df = adfuller(resample_df.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(test_df[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
for key, value in test_df[4].items():
  df_output['Critical Value (%s)'%key] = value
print(df_output)

decomp = seasonal_decompose(resample_df)
trend = decomp. trend
seasonal = decomp.seasonal
residual = decomp.resid

plt.subplot (411)
plt.plot(resample_df)
plt. xlabel('Original')
plt.figure(figsize=(6,5))
plt.subplot(412)
plt.plot (trend)
plt.xlabel('Trend')
plt.figure(figsize=(6,5))
plt. subplot (413)
plt. plot(seasonal)
plt. xlabel ( 'Seasonal')
plt.figure(figsize=(6,5))
plt. subplot (414)
plt. plot(residual)
plt. xlabel ('Residual')
plt.figure(figsize=(6,5))
plt. tight_layout()

rol_mean=resample_df.rolling(window=3, center=True).mean ()
ewm=resample_df. ewm(span=3).mean()
rol_std = resample_df.rolling(window=3, center=True).std()
fig, (ax1, ax2)=plt.subplots(1, 2,figsize=(12,5))
ax1. plot(resample_df,label='Original')
ax1. plot (rol_mean, label='Rolling Mean')
ax1. plot(ewm, label='Exponentially Weighted Mean' )
ax1.set_title( 'Temperature Changes from 1980-2013', fontsize=14)
ax1. set_ylabel ('Temperature',fontsize=12)
ax1.set_xlabel( 'Year' ,fontsize=12)
ax1. legend()
ax2.plot(rol_std,label='Rolling STD')
ax2.set_title('Temperature Changes from 1980-2013',fontsize=14)
ax2.set_ylabel ('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend ()
plt.tight_layout()
plt.show()

rol_mean.dropna(inplace=True)
ewm.dropna(inplace=True)
print ('Dickey-Fuller Test for the Rolling Mean:')
df_test=adfuller(rol_mean.iloc[:,0].values, autolag='AIC')
df_output=pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
for key, value in df_test[4].items():
  df_output['Critical Value (%s) '%key]=value
print (df_output)
print ('')
print ('Dickey-Fuller Test for the Exponentially Weighted Mean:')
df_test = adfuller(ewm.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used' ])
for key, value in df_test[4].items():
  df_output[ 'Critical Value (%s) '%key]=value
print(df_output)

diff_rol_mean = resample_df - rol_mean
diff_rol_mean.dropna(inplace=True)
diff_rol_mean.head()

diff_ewm = resample_df - ewm
diff_ewm.dropna(inplace=True)
diff_ewm.head()

df_rol_mean_diff=diff_rol_mean.rolling(window=3, center=True).mean()
df_ewm_diff = diff_ewm.ewm(span=3).mean()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.plot(diff_rol_mean, label='Original')
ax1.plot(df_rol_mean_diff,label='Rolling Mean')
ax1.set_title( 'Temperature Changes from 1980-2013', fontsize=14)
ax1.set_ylabel ( 'Temperature', fontsize=12)
ax1.set_xlabel('Year', fontsize=12)
ax1.legend()
ax2.plot(diff_ewm, label='Original')
ax2.plot(df_ewm_diff,label='Exponentially Weighted Mean')
ax2.set_title( 'Temperature Changes from 1980-2013' ,fontsize=14)
ax2.set_ylabel ('Temperature', fontsize=12)
ax2. set_xlabel('Year', fontsize=12)
ax2. legend()
plt. tight_layout()

print ('Dickey-Fuller Test for the Difference between the Original and Rolling Mean: ')
dftest = adfuller(diff_rol_mean.iloc[:,0].values,autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in df_test[4].items():
  dfoutput['Critical Value (%s) '%key]=value
print (dfoutput)
print ('')
print ('Dickey-Fuller Test for the Exponentially Weighted Mean:')
dftest = adfuller(diff_ewm.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used' ])
for key, value in dftest[4].items():
  dfoutput[ 'Critical Value (%s) '%key]=value
print(dfoutput)
