# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:58:34 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plastic=pd.read_csv('C:/Users/HP/Desktop/assignments submission/forecasting/PlasticSales.csv')
plastic['months']=0


for i in range(60):
    p=plastic['Month'][i]
    plastic['months'][i]=p[0:3]

month_dummies=pd.get_dummies(plastic['months'])
Plastic=pd.concat([plastic,month_dummies],axis=1)
Plastic['t']=np.arange(1,61)
Plastic['t_sq']=Plastic['t']*Plastic['t']
Plastic['log_Sales']=np.log(Plastic['Sales'])

Train=Plastic[0:40]
Test=Plastic[40:]
plt.plot(Plastic.iloc[:,1])
Test.set_index(np.arange(1,21),inplace=True)

########## Linear ##############
import statsmodels.formula.api as smf
lin_model=smf.ols('Sales~t',data=Train).fit()
predict_lin=lin_model.predict(Test['t'])
error_lin=Test['Sales']-predict_lin
rmse_lin=np.sqrt(np.mean(error_lin**2))
rmse_lin# 248.92


############ Exponential ############
exp_model=smf.ols('log_Sales~t',data=Train).fit()
predict_exp=exp_model.predict(Test['t'])
error_exp=Test['Sales']-predict_exp
rmse_exp=np.sqrt(np.mean(error_exp**2))
rmse_exp# 1380.83


########## Quadratic ##############
import statsmodels.formula.api as smf
quad_model=smf.ols('Sales~t+t_sq',data=Train).fit()
predict_quad=quad_model.predict(Test[['t','t_sq']])
error_quad=Test['Sales']-predict_quad
rmse_quad=np.sqrt(np.mean(error_quad**2))
rmse_quad# 495.46

########## Additive Seasonality ##############
import statsmodels.formula.api as smf
add_sea_model=smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
predict_add_sea=add_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_add_sea=Test['Sales']-predict_add_sea
rmse_add_sea=np.sqrt(np.mean(error_add_sea**2))
rmse_add_sea# 263.23

######### Additive Seasonality Quadratic ###########
add_sea_quad_model=smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+t+t_sq',data=Train).fit()
predict_add_sea_quad=add_sea_quad_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']])
error_add_sea_quad=Test['Sales']-predict_add_sea_quad
rmse_add_sea_quad=np.sqrt(np.mean(error_add_sea_quad**2))
rmse_add_sea_quad# 118.23


##########  Mutiplicative Seasonality ##################
import statsmodels.formula.api as smf
mul_sea_model=smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
predict_mul_sea=mul_sea_model.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_mul_sea=Test['Sales']-predict_mul_sea
rmse_mul_sea=np.sqrt(np.mean(error_mul_sea**2))
rmse_mul_sea# 1380.93

########### Multiplicative Additive Seasonality #################
import statsmodels.formula.api as smf
mul_add_sea_model=smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
predict_mul_add_sea=mul_add_sea_model.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
error_mul_add_sea=Test['Sales']-predict_mul_add_sea
rmse_mul_add_sea=np.sqrt(np.mean(error_mul_add_sea**2))
rmse_mul_add_sea# 1380.72

data={'model':['lin_model','exp_model','quad_model','add_sea','add_sea_quad','mul_sea','mul_add_sea'],'rmse_val':[rmse_lin,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add_sea]}
rmse_table=pd.DataFrame(data)
rmse_table

#Additive Seasonality Quadratic is having least rmse
#So Additive Seasonality Quadratic model is the best model

