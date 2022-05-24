# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:26:09 2022

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as svc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 
train. drop("row_id", axis=1, inplace=True)
test_rows=test['row_id']
test. drop("row_id", axis=1, inplace=True)


#%%
train=train.drop_duplicates()

#%%
target_encoder = LabelEncoder()
target_encoder.fit(train['target'])
target_values = target_encoder.transform(train['target'])

target_unique_values=train["target"].unique()

target=train['target']
train. drop("target", axis=1, inplace=True)


#%%

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train = ss.fit_transform(train)

#%%

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)
svm = svc(kernel='rbf', C = 50)

svm.fit(X_train, y_train)
y_pred_test = svm.predict(X_test)
print('Validation Accuracy: ',metrics.accuracy_score(y_test, y_pred_test))
print("Precision: ", metrics.precision_score(y_test, y_pred_test, average='macro'))
print("Recall: ", metrics.recall_score(y_test, y_pred_test, average='macro'))
print("F1_score: ", metrics.f1_score(y_test, y_pred_test, average='macro'))
y_pred_train = svm.predict(X_train)
print("Training Accuracy: ", metrics.accuracy_score(y_train, y_pred_train))

test = ss.fit_transform(test)
y_pred = svm.predict(test)
y_pred = target_encoder.inverse_transform(y_pred)
y_pred_train= target_encoder.inverse_transform(y_pred_train)
print("\nPredicted target values: ",y_pred)

svm_submission = pd.DataFrame({"row_id":test_rows, "target":y_pred})
svm_submission.to_csv("svm_submission_2.csv", index = False)

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
from sklearn.svm import SVC
import matplotlib.pyplot as plt

classes=target_unique_values
target = label_binarize(list(target), classes=classes)
n_classes =  10

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =train_test_split(train, target, test_size=0.2, random_state=0)

# classifier
clf = OneVsRestClassifier(SVC(kernel='rbf', C = 50))
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
