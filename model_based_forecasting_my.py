# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:33:10 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

amtrak = pd.read_csv("C:/Users/HP/Desktop/python prgrmg/forecasting/Amtrak.csv")
month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
amtrak['Month'][0]
amtrak['months']=0


for i in range(159):
    p=amtrak['Month'][i]
    amtrak['months'][i]=p[0:3]
    
amtrak.rename(columns={"Ridership ('000)":"Ridership"},inplace=True)
month_dummies=pd.get_dummies(amtrak['months'])
Amtrak=pd.concat([amtrak,month_dummies],axis=1)
Amtrak['t']=np.arange(1,160)
Amtrak['t_sq']=Amtrak['t']*Amtrak['t']
Amtrak['log_Rider']=np.log(Amtrak['Ridership'])

Train=Amtrak[0:147]
Test=Amtrak[147:]
plt.plot(Amtrak.iloc[:,1])
Test.set_index(np.arange(1,13),inplace=True)


######## Linear ##########
import statsmodels.formula.api as smf
lin_model=smf.ols('Ridership~t',data=Train).fit()
pred_lin=lin_model.predict(Test['t'])
Error_lin=Test['Ridership']-pred_lin
rmse_lin=np.sqrt(np.mean(Error_lin**2))
rmse_lin#209.92


######### Exponential ###############
import statsmodels.formula.api as smf
exp_model=smf.ols('log_Rider~t',data=Train).fit()
pred_exp=exp_model.predict(Test['t'])
Error_exp=Test['Ridership']-pred_exp
rmse_exp=np.sqrt(np.mean(Error_exp**2))
rmse_exp#2062.95

 
########## Quadratic ################
import statsmodels.formula.api as smf
quad_model=smf.ols('Ridership~t+t_sq',data=Train).fit()
pred_quad=quad_model.predict(Test[["t","t_sq"]])
Error_quad=Test['Ridership']-pred_quad
rmse_quad=np.sqrt(np.mean(Error_quad**2))
rmse_quad#137.154

######## Additive Seasonality  ############
add_sea_model=smf.ols('Ridership~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea=add_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
Error_add_sea=Test['Ridership']-pred_add_sea
rmse_add_sea=np.sqrt(np.mean(Error_add_sea**2))
rmse_add_sea#264.66


############ Additive Seasonality Quadratic ########
add_sea_quad_model=smf.ols('Ridership~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+t+t_sq',data=Train).fit()
pred_add_sea_quad=add_sea_quad_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']])
Error_add_sea_quad=Test['Ridership']-pred_add_sea_quad
rmse_add_sea_quad=np.sqrt(np.mean(Error_add_sea_quad**2))
rmse_add_sea_quad#50.60


########## Multiplicative Seasonality #############
mul_sea_model=smf.ols('log_Rider~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mul_sea=mul_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
Error_mul_sea=Test['Ridership']-pred_mul_sea
rmse_mul_sea=np.sqrt(np.mean(Error_mul_sea**2))
rmse_mul_sea#2062.99

########### Multiplicative Additive Seasonality ###########
mul_add_sea_model=smf.ols('log_Rider~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mul_add_sea=mul_add_sea_model.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
Error_mul_add_sea=Test['Ridership']-pred_mul_add_sea
rmse_mul_add_sea=np.sqrt(np.mean(Error_mul_add_sea**2))
rmse_mul_add_sea#2062.9

##################### Testing ########################
data={'Model':['rmse_lin','rmse_exp','rmse_quad','rmse_add_sea','rmse_add_sea_quad','rmse_mul_sea','rmse_mul_add_sea'],'rmse_val':[rmse_lin,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add_sea]}
table_rmse=pd.DataFrame(data)
table_rmse

# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

predict_data=pd.read_excel('C:/Users/HP/Desktop/python prgrmg/forecasting/predict_new.xlsx')
final_model=smf.ols('Ridership~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+t+t_sq',data=Amtrak).fit()
predictions=final_model.predict(predict_data)
predictions
predict_data['forecasted_data']=predictions
predict_data['forecasted_data'].plot
