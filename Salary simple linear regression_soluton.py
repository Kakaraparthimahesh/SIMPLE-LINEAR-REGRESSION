# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:27:00 2022

@author: MAHESH
"""
#2) Salary_hike -> Build a prediction model for Salary_hike

# step :1 ---> import the data set

import pandas as pd
df=pd.read_csv("Salary_Data.csv")
list(df)
df.shape

# step : 2 ------> Split the variables as x and y

X = df[["YearsExperience"]]

Y = df["Salary"]

# step :3----> plot the scatter plot between x and y's

import matplotlib.pyplot as plt
plt.scatter(X,Y,color="black")
plt.show()

df.corr()

#y=-mx+c
#=====================================================

# step : 4------> model fitting 

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
 
LR.intercept_            # BO = 25792.20019867
LR.coef_                 # B1 = 9449.96232146

# step : 5 ------> predicating the values

Y_pred=LR.predict(X)

# SCATTER PLOT
import matplotlib.pyplot as plt
plt.scatter(X,Y,color ="black")
plt.scatter(X,Y_pred,color="Red")
plt.show()

# list of the variable names with the data type it iS
df.info()     

df.isnull().sum()   # finding missing values

#  **EXPLORATORY OF DATA ANALYSIS**

# HISTOGRAM

df.hist("YearsExperience")  

from  scipy.stats import kurtosis,skew

kurtosis(df["YearsExperience"],fisher=False)
skew(df["YearsExperience"])     

# BOX PLOT
df.boxplot("YearsExperience")

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


#  MSE  =   31270951.722280957
#  RMSE =   5592.04
#  r2   =   95.7