# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:40:51 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
walmart=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/forecasting/footfalls.csv')
walmart.Footfalls.plot()

Train=walmart[0:147]
Test=walmart[147:]


#Moving Average

plt.figure(figsize=(12,4))
walmart.Footfalls.plot(label='org')
for i in range(2,24,6):
    walmart["Footfalls"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

#Time series decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(walmart.Footfalls,period=12)
decompose_ts_add.plot()
plt.show()


#ACF and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(walmart.Footfalls,lags=12)
tsa_plots.plot_pacf(walmart.Footfalls,lags=12)
plt.show()

#MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

######### Simple Exponential Model ##############

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train["Footfalls"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
ses_mape=MAPE(pred_ses,Test.Footfalls) 
ses_mape #8.49

########## Holt's method or Double Exponential #############

from statsmodels.tsa.holtwinters import Holt
hw_model = Holt(Train["Footfalls"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
hw_mape=MAPE(pred_hw,Test.Footfalls) 
hw_mape #7.54

######## Holts winter exponential smoothing with additive seasonality and additive trend #########

from statsmodels.tsa.holtwinters import ExponentialSmoothing 
hwe_model_add_add = ExponentialSmoothing(Train["Footfalls"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
hw_add_add_mape=MAPE(pred_hwe_add_add,Test.Footfalls) 
hw_add_add_mape #2.77

######### Holts winter exponential smoothing with multiplicative seasonality and additive trend ######

hwe_model_mul_add = ExponentialSmoothing(Train["Footfalls"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
hw_mul_add_mape=MAPE(pred_hwe_mul_add,Test.Footfalls) 
hw_mul_add_mape #3.99

data={'Model':['ses','hw','hw_add_add','hw_mul_add'],'MAPE_val':[ses_mape,hw_mape,hw_add_add_mape,hw_mul_add_mape]}
table_MAPE=pd.DataFrame(data)
table_MAPE


# As Holts winter exponential smoothing with additive seasonality and additive trend has less MAPE
# Build Final Model by combining train and test

hwe_model_add_add = ExponentialSmoothing(walmart["Footfalls"],seasonal="add",trend="add",seasonal_periods=12).fit()

#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)
