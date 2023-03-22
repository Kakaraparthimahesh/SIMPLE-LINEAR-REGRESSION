# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:42:19 2022

@author: MAHESH
"""
#1) Delivery_time -> Predict delivery time using sorting time 

# step :1 ---> import the data set

import pandas as pd
df=pd.read_csv("delivery_time.csv")
df
df.shape
list(df)

# step : 2 ------> Split the variables as x and y

X = df[["Sorting Time"]]  

Y=df["Delivery Time"]

# step :3----> plot the scatter plot between x and y's

import matplotlib.pyplot as plt
plt.scatter(X,Y,color="black")
plt.show()

df.corr()

#y=-mx+c

#===========================================================
# step : 4------> model fitting 

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
 
LR.intercept_            # BO = 6.58273397
LR.coef_                 # B1 = 1.6490199

# step : 5 ------> predicating the values

Y_pred=LR.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X,Y,color ="black")
plt.scatter(X,Y_pred,color="Red")
plt.show()

# list of the variable names with the data type it iS
df.info()     

df.isnull().sum()   # finding missing values

#  **EXPLORATORY OF DATA ANALYSIS**

# HISTOGRAM

df.hist("Sorting Time")  

from  scipy.stats import kurtosis,skew

kurtosis(df["Sorting Time"],fisher=False)
skew(df["Sorting Time"])     

# BOX PLOT
df.boxplot("Sorting Time")

#calculate error

Y-Y_pred

import numpy as np

# step : 6 ------> calculating mean square error

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
mse

RMSE = np.sqrt(mse)
print('Root Mean square error of above models is:',RMSE.round(2))

R2=r2_score(Y,Y_pred)
print("Rsquare performance of above model is:",(R2*100).round(2))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  MSE  =   7.793311548584062
#  RMSE =   2.79
#  r2   =   68.23

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>









