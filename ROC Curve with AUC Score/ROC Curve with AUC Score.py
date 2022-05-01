# -*- coding: utf-8 -*-
"""
Created on Sun May  1 03:21:08 2022

@author: User
"""

"""ROC Curve with AUC Score"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

iris = datasets.load_iris()

classes=[0,1,2]
target = label_binarize(iris.target, classes=[0,1,2])
n_classes = 3

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =train_test_split(iris.data, target, test_size=0.2, random_state=0)

# classifier
clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', C = 10, probability = True))
y_score = clf.fit(X_train, y_train).decision_function(X_test)
#print(y_score)

# Computing ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#print(y_score[:, 1])
#print(y_test[:, 1])
#print(fpr[1])
#print(tpr[1])

# Plot of a ROC curve for a specific class

for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='AUC score for the class '+str(classes[i])+'= %0.2f' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()