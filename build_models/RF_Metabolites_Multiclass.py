#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:57:46 2022

@author: Talel
"""

#%% Grid Search Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.impute import KNNImputer


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from imblearn.over_sampling import SMOTE


#Load dataset
df = pd.read_csv('MBMulti_normal.csv')


# Train-Test split
X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


# Random Forest model fit
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#scores = cross_val_score(rfc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)



# define search space
space = dict()
space['n_estimators'] = [100, 200, 500]
space['max_features'] = [None, 100, 200]
space['max_depth'] = [None, 50, 75]



# define search
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(rfc, space, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


print("Grid scores on development set:")
print()
means = search.cv_results_["mean_test_score"]
stds = search.cv_results_["std_test_score"]

for mean, std, params in zip(means, stds, search.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
print()


#%% Multiclass ROC Curve

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


rfc_r=RandomForestClassifier(n_estimators = 500, max_depth = 50, max_features = 100)


# fit model
clf = OneVsRestClassifier(rfc_r)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='red', label='Control vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='blue', label='AD vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='green', label='MCI vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);    




#%% Best Model Accuracy

#Best Hyperparameters: {'max_depth': None, 'max_features': 300, 'n_estimators': 500}

df = pd.read_csv('MBMulti_normal.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


# Random Forest model fit
from sklearn.ensemble import RandomForestClassifier
rfc_b=RandomForestClassifier(n_estimators = 500, max_depth = 75, max_features = 100)
rfc_b.fit(X_train, y_train)


 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

#cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv0 = KFold(n_splits=5, random_state=1, shuffle=True)
cv1 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

scores0 = cross_val_score(rfc_b, X_test, y_test, scoring='accuracy', cv=cv0, n_jobs=-1)
scores1 = cross_val_score(rfc_b, X_test, y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
scores2 = cross_val_score(rfc_b, X_test, y_test, scoring='accuracy', cv=cv2, n_jobs=-1)

from numpy import mean
from numpy import std
print('Accuracy: %.3f (%.3f)' % (mean(scores0), std(scores0)))
print('Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))

#pred_rfc=rfc.predict(X_test)
#print(pred_rfc)

#y_pred_train = rfc.predict(X_train)
#print(accuracy_score(y_test,pred_rfc))
#print(accuracy_score(y_train,y_pred_train))


#%% PCA

df = pd.read_csv('MBMulti_normal.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

steps = [('pca', PCA(n_components=300)), ('m', RandomForestClassifier(n_estimators = 500, max_depth = None))]
model = Pipeline(steps=steps)
# evaluate model
cv3 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv3, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


#%% SHAP
import shap


df = pd.read_csv('MBMulti_normal.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


rfc_fi=RandomForestClassifier(n_estimators = 500, max_depth = 50, max_features = 100)
rfc_fi.fit(X_train, y_train)


explainer= shap.TreeExplainer(rfc)
shap_values= explainer.shap_values(X)
shap.summary_plot(shap_values[0], X)



