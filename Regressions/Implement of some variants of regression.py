# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:24:36 2022

@author: User
"""
#%%

"""LINEAR REGRESSION"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

data=pd.read_csv('SimpleLinearRegression.csv')
#print(data)

X=data.YearsExperience.values.reshape(-1,1)
Y=data.Salary.values.reshape(-1,1)

#print('\nX:',X)
#print('\nY:',Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#print(X_train)
#print(X_test)

X_seq = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1,1)
#print(X_seq)

poly = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly.fit(X_train, Y_train)
#print(poly)
linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)

plt.scatter(X_train, Y_train)
plt.plot(X_seq, poly.predict(X_seq), color = "black")
plt.plot(X_seq, linear_regression.predict(X_seq), color = "red")

Y_pred_poly = poly.predict(X_test)
Y_pred_linear = linear_regression.predict(X_test)

#print(Y_pred_poly)
#print(Y_pred_linear)

compare_dataset = pd.DataFrame({'Actual': Y_test.flatten(),
                                'Predicted_Polynomial': Y_pred_poly.flatten(),
                                'Predicted_Linear': Y_pred_linear.flatten()})
print(compare_dataset)

#ERRORS FOR LINEAR REGRESSION
print("\nERRORS FOR LINEAR REGRESSION:")
print('MAE: ', metrics.mean_absolute_error(compare_dataset.Actual,compare_dataset.Predicted_Linear) )
print('MSE: ', metrics.mean_squared_error(compare_dataset.Actual,compare_dataset.Predicted_Linear) )
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(compare_dataset.Actual,compare_dataset.Predicted_Linear)))

#ERRORS FOR Polynomial LINEAR REGRESSION
print("\nERRORS FOR POLYNOMIAL LINEAR REGRESSION:")
print('MAE: ', metrics.mean_absolute_error(compare_dataset.Actual,compare_dataset.Predicted_Polynomial) )
print('MSE: ', metrics.mean_squared_error(compare_dataset.Actual,compare_dataset.Predicted_Polynomial) )
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(compare_dataset.Actual,compare_dataset.Predicted_Polynomial)))

#print(linear_regression.intercept_)
#print(linear_regression.coef_)
ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, Y_train)

Y_pred_ridge = ridge.predict(X_test)
compare_dataset1 = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted_Ridge': Y_pred_ridge.flatten(), 'Predicted_Linear': Y_pred_linear.flatten()})
#print(compare_dataset1)

print("\nErrors for Ridge rgression:")
print('MAE: ', metrics.mean_absolute_error(compare_dataset1.Actual,compare_dataset1.Predicted_Ridge) )
print('MSE: ', metrics.mean_squared_error(compare_dataset1.Actual,compare_dataset1.Predicted_Ridge) )
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(compare_dataset1.Actual,compare_dataset1.Predicted_Ridge)))

#%%

"""LOGISTIC REGRESSION"""

import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('logit_train1.csv', index_col = 0)
X1_train = df[['gmat','gpa','work_experience']]
Y1_train = df[['admitted']]
#print(X1_train)
#print(Y1_train)

logistic = sm.Logit(Y1_train, X1_train).fit()

print(logistic.summary())

#%%
"""LOGISTIC REGRESSION"""

"""Testing"""

df = pd.read_csv('logit_test1.csv', index_col = 0)
X1_test = df[['gmat','gpa','work_experience']]
Y1_test = df['admitted']
Y1_pred = logistic.predict(X1_test)
#print(Y1_pred)

prediction = list(map(round,Y1_pred))
#print(prediction)

print('Actual class:    ', list(Y1_test.values))
print('Predicted class: ', prediction)

#%%

"""Confusion Matrix"""

"""
False Negative (FN): Actual class = 1, Predicted class = 0

False Positive (FP): Actual class = 0, Predicted class = 1

True Positive (TP): Actual class = 1, Predicted class = 1

True Negative (TN): Actual class = 0, Predicted class = 0

Classification Evaluation Metrics

Accuracy = TP+TN / TP+TN+FP+FN

Recall = TP / TP + FN (The percentage of correctly positively identified samples among all actually positive samples)

Precision = TP / TP + FP (The percentage of correctly positively identified samples among all predicted positive samples)

F-meansure = harmonic mean pf precision and recall
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

print("\nConfusion matrix for sklearn librery:")
cm = confusion_matrix (Y1_test, prediction)
print(cm)

print(accuracy_score(Y1_test, prediction))

logistic_reg = LogisticRegression(C = 0.1) #C is inverse of regularization strength.
logistic_reg.fit(X1_train, Y1_train)

prediction_log_reg = logistic_reg.predict(X1_test)
print('Actual class:                       ', np.array(Y1_test.values))
print('Predicted class (using sklearn):    ', prediction_log_reg)
print('Predicted class (using statsmodel): ', np.array(prediction))

print("\nConfusion matrix for Statsmodel librery:")
cm1 = confusion_matrix (Y1_test, prediction_log_reg)
print(cm1)

print(accuracy_score(Y1_test, prediction_log_reg))

#print(logistic_reg.intercept_)
#print(logistic_reg.coef_)