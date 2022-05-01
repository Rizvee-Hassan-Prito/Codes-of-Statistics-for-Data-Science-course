# -*- coding: utf-8 -*-
"""
Created on Sun May  1 07:44:01 2022

@author: User
"""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target,
                                      test_size = 0.2, random_state = 42)

model = svm.SVC(kernel = 'linear', C = 10, probability = True)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
compare_dataset = pd.DataFrame({'Actual': Y_test.flatten(), 
                                'Predicted': Y_pred.flatten()})

print("\nFor SVM:")
print("\nConfusion Matrix:")
cm = metrics.confusion_matrix (Y_test, Y_pred)
print(cm)
print()
print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))
print("Precision: ", metrics.precision_score(Y_test, Y_pred, average='macro'))
print("Recall: ", metrics.recall_score(Y_test, Y_pred, average='macro'))

y_pred_train = model.predict(X_train)

print("Training Accuracy: ", metrics.accuracy_score(Y_train, y_pred_train))

#%%
print("\n\nFor Logistic Regression:")
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression(C = 10, solver='newton-cg')

logistic_reg.fit(X_train, Y_train)
prediction_log_reg = logistic_reg.predict(X_test)

print("\nConfusion Matrix:")
cm = metrics.confusion_matrix (Y_test, prediction_log_reg)
print(cm)
print()

print("Accuracy: ", metrics.accuracy_score(Y_test, prediction_log_reg))
print("Precision: ", metrics.precision_score(Y_test, prediction_log_reg,average='macro'))
print("Recall: ", metrics.recall_score(Y_test, prediction_log_reg, average='macro'))

logistic_Y_pred_train = logistic_reg.predict(X_train)
print("Training Accuracy: ", metrics.accuracy_score(Y_train, logistic_Y_pred_train))
print()
#%%

"""ROC Curve with AUC Score"""

from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.preprocessing import label_binarize

classes=[0,1,2]
target = label_binarize(iris.target, classes=[0,1,2])
n_classes = 3

# shuffling and spliting training and test sets
X_train, X_test, y_train, y_test =train_test_split(iris.data, target, test_size=0.2, random_state=0)

# classifier
clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', C = 10, probability = True))
y_score = clf.fit(X_train, y_train).decision_function(X_test)


# Computing ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Ploting ROC curve for a specific class

for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='AUC score for the class '+str(classes[i])+'= %0.2f' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()