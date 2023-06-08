#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:57:46 2022

@author: Talel
"""

#%% Grid Search

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
df = pd.read_csv('DatasetPTMCI.csv')


# Train-Test split
X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


# Logistic Regression model fit
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#scores = cross_val_score(lr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)



# parameter grid
parameters = {
    'penalty' : ['l2'], 
    'C'       : [0.5, 1],
    'solver'  : ['newton-cg', 'lbfgs', 'sag']}


# define search
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(lr, param_grid = parameters, scoring='accuracy', cv=cv)                     


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

#%% Grid Search 2

#Load dataset 
df = pd.read_csv('DatasetPTMCI.csv')



# Train-Test split
X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


#  Logistic Regression model fit
from sklearn.linear_model import LogisticRegression

lr2=LogisticRegression(max_iter=5000)
lr2.fit(X_train, y_train)


# parameter grid
parameters_bis = {
    'penalty' : ['l1'], 
    'C'       : [1],
    'solver'  : ['saga']}


# define search
from sklearn.model_selection import GridSearchCV
                  

search_bis = GridSearchCV(lr2, param_grid = parameters_bis, scoring='accuracy', cv=cv) 

# execute search
result_bis = search_bis.fit(X, y)
# summarize result
print('Best Score: %s' % result_bis.best_score_)
print('Best Hyperparameters: %s' % result_bis.best_params_)


print("Grid scores on development set:")
print()
means_bis = search_bis.cv_results_["mean_test_score"]
stds_bis = search_bis.cv_results_["std_test_score"]

for mean, std, params in zip(means_bis, stds_bis, search_bis.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
print()


#%% Best Model Accuracy



df = pd.read_csv('DatasetPTMCI.csv')

X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape



# Logistic Regression model fit
from sklearn.linear_model import LogisticRegression
lr_b=LogisticRegression(max_iter=5000, C=1, penalty ='l2', solver='lbfgs')
lr_b.fit(X_train, y_train)


 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

#cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv0 = KFold(n_splits=5, random_state=1, shuffle=True)
cv1 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

scores0 = cross_val_score(lr_b, X_test, y_test, scoring='accuracy', cv=cv0, n_jobs=-1)
scores1 = cross_val_score(lr_b, X_test, y_test, scoring='accuracy', cv=cv1, n_jobs=-1)
scores2 = cross_val_score(lr_b, X_test, y_test, scoring='accuracy', cv=cv2, n_jobs=-1)



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

df = pd.read_csv('DatasetPTMCI.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


lr_p=LogisticRegression(max_iter=1000, C=1, penalty ='l2', solver='lbfgs')
lr_p.fit(X_train, y_train)


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


steps = [('pca', PCA(n_components=300)), ('m', lr_p)]
model = Pipeline(steps=steps)
# evaluate model
cv3 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv3, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% Feature Importance


df = pd.read_csv('DatasetPTMCI.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape



# SVM model fit
from sklearn.linear_model import LogisticRegression
lr_fi=LogisticRegression(max_iter=1000, C=1, penalty ='l2', solver='lbfgs')
lr_fi.fit(X_train, y_train)


from sklearn.inspection import permutation_importance
results = permutation_importance(lr_fi, X, y, scoring='neg_mean_squared_error')    
importance = results.importances_mean

#for i, v in enumerate(importance):
#    print('Feature: %0d, Score: %.5f' % (i,v))
    
df_importance = pd.DataFrame(importance)
df_importance.to_csv('FI_LRBestMulti')


#pd.Series(abs(lr_fi.coef_[-1]), index=X.columns).nlargest(20).plot(kind='barh').invert_yaxis()
#pd.Series(abs(lr_fi.coef_[-1]), index=X.columns).nlargest(20).plot.barh(stacked=True)



#%% Multiclass ROC Curve

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# SVM model fit
from sklearn.linear_model import LogisticRegression
lr_fi=LogisticRegression(max_iter=1000, C=1, penalty ='l2', solver='lbfgs')


# fit model
clf = OneVsRestClassifier(lr_fi)
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
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='MCI vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='AD vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);    

