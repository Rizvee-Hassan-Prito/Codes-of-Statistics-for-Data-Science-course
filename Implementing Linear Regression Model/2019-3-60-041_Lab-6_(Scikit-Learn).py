# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 06:17:20 2022

@author: User
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

print('\nEnter file location/path:')
file_name=input()
data = pd.read_csv(file_name)

print('\nEnter columns:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break

data.columns=columns
#data.columns = ['Country','Newspapers','Radios','TVsets','literacyRate']

print("\nGiven dataset:")
print(data)

print("\nGiven dataset:")
print(data)

# Preparing X (independent variables) and Y (dependent variables)

print('\nEnter independent variable names:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break

X = data[columns]
print('\nEnter dependent variable names:')
columns=[]
while True:
    ln=input()
    if ln:
        columns.append(ln)
    else:
        break

Y = data[columns]

print("\nIndependent Variables:\n")
print(X)
print("\nDependent Variable:\n")
print(Y)

# splitting dataset into train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Fitting the model 

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print("\nIntercept:",regressor.intercept_)
print("\nCoefficients:",regressor.coef_)

# predicting y for test dataset

Y_pred = regressor.predict(X_test)
#print("\nPredicted values of Y:")
#print(Y_pred)


# comparing result
y_test = np.asarray(Y_test)
y_pred = np.asarray(Y_pred)
compare_result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print("\nComparing actual values of Y and predicted values of Y:")
print(compare_result)

# Evaluating Metrics
print('MAE: ', metrics.mean_absolute_error(compare_result.Actual,compare_result.Predicted) )
print('MSE: ', metrics.mean_squared_error(compare_result.Actual,compare_result.Predicted) )
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(compare_result.Actual,compare_result.Predicted)))
print('R-squared: ', metrics.r2_score(compare_result.Actual,compare_result.Predicted))