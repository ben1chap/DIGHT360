# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:48:28 2018

@author: Ben
"""

import pandas
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE  # Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('mc_feat_names.txt') as name_file:
    names = name_file.read().strip().split('\t')
len_names = len(names)
with open('mc_features.csv') as mc_file:
    dataset = pandas.read_csv(mc_file, names=names,  # pandas DataFrame object
                              keep_default_na=False, na_values=['_'])  # avoid 'NA' category being interpreted as missing data  # noqa
print(list(dataset))  # easy way to get feature (column) names

LRmodel = LogisticRegression()
array = dataset.values  # numpy array
feats = array[:,0:len_names - 1]  # to understand comma, see url in next line:
labels = array[:,-1]  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing

ETCmodel = ExtraTreesClassifier()
ETCmodel.fit(feats, labels)
print('importances', ETCmodel.feature_importances_)

rfe = RFE(LRmodel, 3)
rfe = rfe.fit(feats, labels)

predictions = rfe.predict(feats)


print(list(dataset))
print(rfe.support_)
print(rfe.ranking_)
print('Accuracy', accuracy_score(labels, predictions))

