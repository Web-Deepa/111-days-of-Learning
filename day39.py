#Time Series Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

#1.generate data
np.random.seed(42)
dates=pd.date_range('2025-01-01',periods=100,freq='D')
#create baseline trend ,seasonality,random noise
trend=np.linspace(10,50,100)
seasonality=10 * np.sin(np.linspace(0,3.14 * 2,100))
noise=np.random.normal(0,2,100)
values=trend + seasonality + noise

#create dataframe
df=pd.DataFrame({'Value':values},index=dates)
print(df)

# 2. Visualize the Data
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Original Data')
plt.title('Original Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig("timedata.png")
plt.show()

#3.Seasonal Decomposition
decom=sm.tsa.seasonal_decompose(df['Value'],model='additive')
fig=decom.plot()
fig.set_size_inches(12,8)
plt.savefig("seasonal.png")
plt.show()

#4.Stationary Test
def check_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest=adfuller(timeseries,autolag='AIC')
    dfout=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags used','Number of Observations used'])
    for key,value in dftest[4].items():
        dfout['Critical value(%s)'%key]=value
    print(dfout)
check_stationarity (df['Value'])

#5.ARIMA Modeling
tr=df.iloc[:-10]
te=df.iloc[-10]
model=ARIMA(tr,order=(1,1,1))
model_fit=model.fit()
print(model_fit.summary())

#6.Forecasting
preds=model.fit.forecast(steps=10)
preds.index=te.index
plt.figure(figsize=(12,5))
plt.plot(tr.index,tr['Value'],label='Train Data')
plt.plot(te.index,te['Value'],label=' Actual Test Data')
plt.plot(preds.index,preds,label='ARIMA Forecast',color='red',linestyle='---')
plt.title('Train ,Test and Forecasted Values' )
plt.xlabel("Date")
plt.ylabel('Value')
plt.legend()
plt.savefig("fore.png")
plt.show()