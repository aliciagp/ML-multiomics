#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:57:46 2022

@author: Talel
"""

#%% Grid Search SVM

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


# SVM model fit
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train, y_train)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#scores = cross_val_score(rfc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


# define search space

# Kernel Type = Linear
print('----> Kernel = Linear')

space = dict()
space['kernel'] = ['linear']
space['C'] = [1, 5, 10, 20, 50]
space['decision_function_shape'] = ['ovo', 'ovr']


# define search
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(svm, space, scoring='accuracy', n_jobs=-1, cv=cv)
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

#%%

# Kernel Type = Gaussian
print('----> Kernel = Gaussian')

spaceG = dict()
spaceG['kernel'] = ['rbf']
spaceG['C'] = [10, 20, 50]
#spaceG['gamma'] = ['scale', 1e-5, 1e-10]
spaceG['decision_function_shape'] = ['ovo', 'ovr']


# define search
from sklearn.model_selection import GridSearchCV
searchG = GridSearchCV(svm, spaceG, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = searchG.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


print("Grid scores on development set:")
print()
meansG = searchG.cv_results_["mean_test_score"]
stdsG = searchG.cv_results_["std_test_score"]

for mean, std, params in zip(meansG, stdsG, searchG.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
print()

#%%

# Kernel Type = Polynomial
print('----> Kernel = Polynomial')

spaceP = dict()
spaceP['kernel'] = ['poly']
spaceP['C'] = [10, 20, 50]
spaceP['degree'] = [2, 3, 4]
#spaceP['gamma'] = ['scale', 1e-5, 1e-10]
spaceP['decision_function_shape'] = ['ovo', 'ovr']


# define search
from sklearn.model_selection import GridSearchCV
searchP = GridSearchCV(svm, spaceP, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = searchP.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


print("Grid scores on development set:")
print()
meansP = searchP.cv_results_["mean_test_score"]
stdsP = searchP.cv_results_["std_test_score"]

for mean, std, params in zip(meansP, stdsP, searchP.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
print()


#%%

# Kernel Type = Sigmoid
print('----> Kernel = Sigmoid')

spaceS = dict()
spaceS['kernel'] = ['sigmoid']
spaceS['C'] = [1, 5, 10, 20, 50]
spaceS['gamma'] = ['scale', 1e-5, 1e-10]
spaceS['decision_function_shape'] = ['ovo', 'ovr']


# define search
from sklearn.model_selection import GridSearchCV
searchS = GridSearchCV(svm, spaceS, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = searchS.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


print("Grid scores on development set:")
print()
meansS = searchS.cv_results_["mean_test_score"]
stdsS = searchS.cv_results_["std_test_score"]

for mean, std, params in zip(meansS, stdsS, searchS.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
print()




#%% Feature Importance

from sklearn.inspection import permutation_importance


#----> Kernel = Linear
#Best Score: 0.8724946509429269
#Best Hyperparameters: {'C': 1, 'decision_function_shape': 'ovo', 'kernel': 'linear'}

#----> Kernel = Gaussian
#Best Score: 0.8840075633179082
#Best Hyperparameters: {'C': 20, 'decision_function_shape': 'ovo', 'gamma': 'scale', 'kernel': 'rbf'}

#----> Kernel = Polynomial
#Best Hyperparameters: {'C': 10, 'decision_function_shape': 'ovo', 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
#Best Score: 0.8819948250982733



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


# SVM Linear kernel
from sklearn.svm import SVC
svm_L=SVC(C = 1, decision_function_shape = 'ovo', kernel = 'linear')
svm_L.fit(X_train, y_train)


pd.Series(abs(svm_L.coef_[-1]), index=X.columns).nlargest(20).plot(kind='barh').invert_yaxis()
#pd.Series(abs(svm.coef_[-1]), index=X.columns).nlargest(20).plot.barh(stacked=True)

#%%
# SVM Gaussian kernel
from sklearn.svm import SVC
svm_G=SVC(C = 20, decision_function_shape = 'ovo', gamma = 'scale', kernel = 'rbf')
svm_G.fit(X_train, y_train)


results_G = permutation_importance(svm_G, X, y, scoring='neg_mean_squared_error')    
importance_G = results_G.importances_mean

for i, v in enumerate(importance_G):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
df_importG = pd.DataFrame(importance_G)
df_importG.to_csv('Feature_import_SVMGaussian')



#%%
#SVM Polynomial kernel
from sklearn.svm import SVC
svm_P=SVC(C = 5, decision_function_shape = 'ovo',  degree = 3, gamma = 'scale', kernel = 'poly')
svm_P.fit(X_train, y_train)


results_P = permutation_importance(svm_P, X, y, scoring='neg_mean_squared_error')    
importance_P = results_P.importances_mean

for i, v in enumerate(importance_P):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
df_importP = pd.DataFrame(importance_P)
df_importP.to_csv('Feature_import_SVMPoly')





#%% Best Model Accuracy

#Best Hyperparameters: {'max_depth': None, 'max_features': 300, 'n_estimators': 500}

df = pd.read_csv('DatasetPTMCI.csv')


X = df.drop('Diagnosis',axis=1)
y = df['Diagnosis']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
X_train.shape, X_test.shape


# SVM
from sklearn.svm import SVC
svm_b=SVC(C = 20, decision_function_shape = 'ovo', kernel = 'rbf', gamma = 'scale')
svm_b.fit(X_train, y_train)


 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

#cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv0 = KFold(n_splits=5, random_state=1, shuffle=True)
cv1 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

scores0 = cross_val_score(svm_b, X, y, scoring='accuracy', cv=cv0, n_jobs=-1)
scores1 = cross_val_score(svm_b, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)
scores2 = cross_val_score(svm_b, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)

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




#%% Multiclass ROC Curve

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


svm_r=SVC(C = 20, decision_function_shape = 'ovo', kernel = 'rbf', gamma = 'scale', probability=True)


# fit model
clf = OneVsRestClassifier(svm_r)
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

