# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:30:09 2022

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC as svc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

train = pd.read_csv("train.csv")
train. drop("row_id", axis=1, inplace=True)

train=train.drop_duplicates()
target_encoder = LabelEncoder()
target_encoder.fit(train['target'])
target_values = target_encoder.transform(train['target'])

target_unique_values=train["target"].unique()

target=train['target']
train. drop("target", axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train = ss.fit_transform(train)

#%%
X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)

#%%
log_reg = LogisticRegression(C = 1 , solver='liblinear')

log_reg.fit(X_train, y_train)
y_pred_test = log_reg.predict(X_test)

print("\nFor Logistic Regression:\n")

t_accuracy_score_lg = metrics.accuracy_score(y_test, y_pred_test)
print('Validation Accuracy: ',t_accuracy_score_lg)

precision_score_lg= metrics.precision_score(y_test, y_pred_test, average='macro')
print("Precision: ",precision_score_lg)

recall_score_lg = metrics.recall_score(y_test, y_pred_test, average='macro')
print("Recall: ", recall_score_lg)

f1_score_lg = metrics.f1_score(y_test, y_pred_test, average='macro')
print("F1_score: ", f1_score_lg)

y_pred_train = log_reg.predict(X_train)
tr_accuracy_score_lg = metrics.accuracy_score(y_train, y_pred_train)
print("Training Accuracy: ", tr_accuracy_score_lg)

#%%
log_reg = LogisticRegression(C = 0.1, solver='newton-cg')

log_reg.fit(X_train, y_train)
y_pred_test = log_reg.predict(X_test)

print("\nFor Upgraded Logistic Regression:\n")

t_accuracy_score_lg_2 = metrics.accuracy_score(y_test, y_pred_test)
print('Validation Accuracy: ',t_accuracy_score_lg_2)

precision_score_lg_2= metrics.precision_score(y_test, y_pred_test, average='macro')
print("Precision: ",precision_score_lg_2)

recall_score_lg_2 = metrics.recall_score(y_test, y_pred_test, average='macro')
print("Recall: ", recall_score_lg_2)

f1_score_lg_2 = metrics.f1_score(y_test, y_pred_test, average='macro')
print("F1_score: ", f1_score_lg_2)

y_pred_train = log_reg.predict(X_train)
tr_accuracy_score_lg_2 = metrics.accuracy_score(y_train, y_pred_train)
print("Training Accuracy: ", tr_accuracy_score_lg_2)

#%%

svm = LinearSVC(C = 1 , random_state=42)

svm.fit(X_train, y_train)
y_pred_test = svm.predict(X_test)

print("\nFor SVM:\n")

t_accuracy_score_svm = metrics.accuracy_score(y_test, y_pred_test)
print('Validation Accuracy: ',t_accuracy_score_svm)

precision_score_svm= metrics.precision_score(y_test, y_pred_test, average='macro')
print("Precision: ",precision_score_svm)

recall_score_svm = metrics.recall_score(y_test, y_pred_test, average='macro')
print("Recall: ", recall_score_svm)

f1_score_svm = metrics.f1_score(y_test, y_pred_test, average='macro')
print("F1_score: ", f1_score_svm)

y_pred_train = svm.predict(X_train)
tr_accuracy_score_svm = metrics.accuracy_score(y_train, y_pred_train)
print("Training Accuracy: ", tr_accuracy_score_svm)

#%%

svm = svc(kernel='rbf', C = 50)

svm.fit(X_train, y_train)
y_pred_test = svm.predict(X_test)

print("\nFor Upgraded SVM:\n")

t_accuracy_score_svm_2 = metrics.accuracy_score(y_test, y_pred_test)
print('Validation Accuracy: ',t_accuracy_score_svm_2)

precision_score_svm_2= metrics.precision_score(y_test, y_pred_test, average='macro')
print("Precision: ",precision_score_svm_2)

recall_score_svm_2 = metrics.recall_score(y_test, y_pred_test, average='macro')
print("Recall: ", recall_score_svm_2)

f1_score_svm_2 = metrics.f1_score(y_test, y_pred_test, average='macro')
print("F1_score: ", f1_score_svm_2)

y_pred_train = svm.predict(X_train)
tr_accuracy_score_svm_2 = metrics.accuracy_score(y_train, y_pred_train)
print("Training Accuracy: ", tr_accuracy_score_svm_2)



#%%

list_t_ac=[]
list_t_ac.append(t_accuracy_score_lg)
list_t_ac.append(t_accuracy_score_lg_2)
list_t_ac.append(t_accuracy_score_svm)
list_t_ac.append(t_accuracy_score_svm_2)

list_prc=[]
list_prc.append(precision_score_lg)
list_prc.append(precision_score_lg_2)
list_prc.append(precision_score_svm)
list_prc.append(precision_score_svm_2)

list_rec=[]
list_rec.append(recall_score_lg)
list_rec.append(recall_score_lg_2)
list_rec.append(recall_score_svm)
list_rec.append(recall_score_svm_2)

list_f1=[]
list_f1.append(f1_score_lg)
list_f1.append(f1_score_lg_2)
list_f1.append(f1_score_svm)
list_f1.append(f1_score_svm_2)

#%%
X = ['Log Reg','Upgraded Log Reg','SVM','Upgraded SVM']
X_axis = np.arange(len(X))
  
plt.bar(X_axis + 0, list_t_ac, color='#FFA600', width = 0.18,label = 'Accuracy')
plt.bar(X_axis + 0.20 , list_prc, color='#FF6361', width = 0.18, label = 'Precision')
plt.bar(X_axis + 0.40, list_rec, color='#BC5090', width = 0.18, label = 'Recall')
plt.bar(X_axis + 0.60, list_f1, color='#58508B', width = 0.18, label = 'F1-score')

 
plt.xticks(X_axis+.30, X)
plt.xlabel("Models",fontsize=12)
plt.ylabel("Scores",fontsize=12)
plt.title("Models' Performance Metrics")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=11, title_fontsize=10)
plt.show()

#%%
list_tr_ac=[]
list_tr_ac.append(tr_accuracy_score_lg)
list_tr_ac.append(tr_accuracy_score_lg_2)
list_tr_ac.append(tr_accuracy_score_svm)
list_tr_ac.append(tr_accuracy_score_svm_2)

  
plt.bar(X_axis + 0.00, list_tr_ac, width = 0.25, label = 'Training Accuracy')
plt.bar(X_axis + 0.25, list_t_ac, width = 0.25, label = 'Validation Accuracy')
 
plt.xticks(X_axis+.11, X, )
plt.xlabel("Models",fontsize=12)
plt.ylabel("Scores",fontsize=12)
plt.title("Validation-Training Accuracy Comparison")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=11, title_fontsize=10)
plt.show()