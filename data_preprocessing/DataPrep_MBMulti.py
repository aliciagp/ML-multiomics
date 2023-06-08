#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:21:33 2020

@author: Talel
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns


#Load dataset as Pandas dataframe
df = pd.read_csv('Metabolomics_Multi.csv')


#Explore dataframe 
print('----> Dataframe Info')
print(df.info())

#print('----> Head')
#print(df.head())

#print('----> Sample')
#print(df.iloc[0,:])

#print('----> Missing values')
#print(df.isnull().sum())



# Data cleaning
df['Diagnosis'] = df['Diagnosis'].map({'MCI': 2, 'NL': 0, 'AD': 1})
df0 = df['Diagnosis']
df.drop('Diagnosis', axis=1, inplace=True)
print('----> Diagnosis removed')
print(df.info())



#KNN nearest neighbours for missing values
imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df)
df1 = pd.DataFrame(data=df_filled)
print('----> Fill blanks')
print(df1.info())


#Data normalization
df2=(df1-df1.min())/(df1.max()-df1.min())

df3 = pd.concat([df0, df2], axis=1)


#Verifying results
print('----> Data Normalisation')

print(df1.head())

print(df2.head())

print(df3.head())


# Reduce dataset from multiclass to binary
#df3 = df3[df3.Diagnosis != 2]
#print(df.info())

#df3 = df3[df3.Diagnosis != 3]
#print(df.info())



#Data shuffling 
df3 = df3.sample(frac=1).reset_index(drop=True)

print('----> Data Shuffling')
print(df3.head())


df3.to_csv('MBMulti_normal.csv')