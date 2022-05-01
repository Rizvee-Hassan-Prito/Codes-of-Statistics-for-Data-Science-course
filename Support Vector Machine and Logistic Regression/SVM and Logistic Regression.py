# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:09:06 2022

@author: User
"""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#print(type(iris))

#print(iris.data)
#print(iris.target)
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target,
                                      test_size = 0.2, random_state = 42)

print("\nFor SVM:")

model = svm.SVC(kernel = 'linear', C = 10, probability = True)
print(Y_test)
###sklearn.svm.svc - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
###Kernel - https://techvidvan.com/tutorials/svm-kernel-functions/#:~:text=A%20kernel%20is%20a%20function,number%20of%20dimensions%20using%20kernels.

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
compare_dataset = pd.DataFrame({'Actual': Y_test.flatten(), 
                                'Predicted': Y_pred.flatten()})

#print(compare_dataset)


"""Metrics"""
print("\nConfusion Metrix:")
cm = metrics.confusion_matrix (Y_test, Y_pred)
print(cm)
print()
print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))
print("Precision: ", metrics.precision_score(Y_test, Y_pred, average='macro'))
print("Recall: ", metrics.recall_score(Y_test, Y_pred, average='macro'))

### metric.precision_score, metric.recall_score -https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.metrics.precision_score.html

y_pred_train = model.predict(X_train)
#print(y_pred_train)
print("Training Accuracy: ", metrics.accuracy_score(Y_train, y_pred_train))


#%%
print("\n\nFor Logistic Regression:")
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression(C = 10, solver='newton-cg')
#sover-https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
logistic_reg.fit(X_train, Y_train)
prediction_log_reg = logistic_reg.predict(X_test)

print("\nConfusion Metrix:")
cm = metrics.confusion_matrix (Y_test, prediction_log_reg)
print(cm)
print()

print("Accuracy: ", metrics.accuracy_score(Y_test, prediction_log_reg))
print("Precision: ", metrics.precision_score(Y_test, prediction_log_reg,average='macro'))
print("Recall: ", metrics.recall_score(Y_test, prediction_log_reg, average='macro'))

logistic_Y_pred_train = logistic_reg.predict(X_train)
print("Training Accuracy: ", metrics.accuracy_score(Y_train, logistic_Y_pred_train))
print()
