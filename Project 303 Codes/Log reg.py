# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:01:22 2022

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 
train. drop("row_id", axis=1, inplace=True)
test_rows=test['row_id']
test. drop("row_id", axis=1, inplace=True)


#%%
"""Data Characteristics and EDA"""

print()
print("\nTrain dataset informations:")
print(train.info())
print("\nGrouped by description of 10 unique target values:")
print(train.groupby("target").describe())

sns.countplot(data = train, y = "target")

"""High Positive Correlations"""

train_cols=list(train.columns)
train_cols.pop(len(train_cols)-1)
col=[]
for i in train_cols:
    for j in train_cols:
        if(train[i].corr(train[j]))>0.7 and i!=j:
            if i not in col:
                col.append(i)
            if j not in col:
                col.append(j)
dict_corr={}
for i in col:
    dict_corr[i]=list(train[i])   
train_corr=pd.DataFrame(dict_corr)

corr = train_corr.corr()
# Setting up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(12, 10))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, annot=True , cmap=cmap, vmax=0.7,vmin=0)

#%%
print("\nNumber of null values in Train dataset:",train.isnull().sum().sum())
print("\nNumber of null values in Test dataset:",test.isnull().sum().sum())

#%%
print("\nNumber of duplicate rows =",list(train.duplicated()).count(True))
train=train.drop_duplicates()

#%%
target_encoder = LabelEncoder()
target_encoder.fit(train['target'])
target_values = target_encoder.transform(train['target'])

print("\nEncoding Target values:\n")
target_unique_values=train["target"].unique()
encoded_unique_values=pd.DataFrame(target_values)[0].unique()
#print(encoded_unique_values)
for i in range(len(target_unique_values)):
    print(encoded_unique_values[i],"=",target_unique_values[i])
print("\n")

target=train['target']
train. drop("target", axis=1, inplace=True)

#%%

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train = ss.fit_transform(train)


#%%

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)

log_reg = LogisticRegression(C = 1 , solver='liblinear')

log_reg.fit(X_train, y_train)
y_pred_test = log_reg.predict(X_test)

print('Validation Accuracy: ',metrics.accuracy_score(y_test, y_pred_test))
print("Precision: ", metrics.precision_score(y_test, y_pred_test, average='macro'))
print("Recall: ", metrics.recall_score(y_test, y_pred_test, average='macro'))
print("F1_score: ", metrics.f1_score(y_test, y_pred_test, average='macro'))
y_pred_train = log_reg.predict(X_train)
print("Training Accuracy: ", metrics.accuracy_score(y_train, y_pred_train))

test = ss.fit_transform(test)
y_pred = log_reg.predict(test)
y_pred = target_encoder.inverse_transform(y_pred)
y_pred_train= target_encoder.inverse_transform(y_pred_train)
print("\nPredicted target values: ",y_pred)

Log_reg_submission = pd.DataFrame({"row_id":test_rows, "target":y_pred})
Log_reg_submission.to_csv("Log_reg_submission.csv", index = False)

#%%
from sklearn.metrics import confusion_matrix

y_test= target_encoder.inverse_transform(y_test)
y_pred_test=target_encoder.inverse_transform(y_pred_test)
CM = confusion_matrix(y_test, y_pred_test, labels=target_unique_values)
print("\n10 unique target values: ",target_unique_values)
print("\nConfusion Matrix for 10 unique target values:\n",CM)
plt.figure(figsize = (10,10))
sns.heatmap(CM/np.sum(CM, axis=0), fmt='.2%', cmap='Reds', annot=True, cbar=True,
            xticklabels=target_unique_values, yticklabels=target_unique_values)
#%%

from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

classes=target_unique_values
target = label_binarize(list(target), classes=classes)
n_classes =  10

# shuffling and spliting training and test sets
X_train, X_test, y_train, y_test =train_test_split(train, target, test_size=0.2, random_state=0)

# classifier
clf = OneVsRestClassifier(LogisticRegression(C = 1 , solver='liblinear'))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Computing ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Ploting of a ROC curve for a specific class
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot([0, 1.0], [0, 1.0], 'k--')
ax.set_xlim([0.0, 2.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves with AUC scores')
for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], label='AUC score for the class '+str(classes[i])+'= %0.2f' % roc_auc[i])
ax.legend(loc="best")
ax.grid(alpha=.4)
sns.despine()
plt.show()
