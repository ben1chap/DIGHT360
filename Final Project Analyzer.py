# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:02:20 2018

@author: Ben
"""
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

analyzer = sia()
files = glob('C:\\Users\\Ben\\Desktop\\F18_DIGHT360\\final-corpus\\*')
pscore = []
stars = []
rank = ['1', '2', '3', '4', '5']
xs = range(len(rank))
for name in files:
    with open(name, encoding='utf8') as reviews:
        for review in reviews:
            star = review[0]
            if star in rank:
                stars.append(star)
                polarity = analyzer.polarity_scores(review)
                pscore.append(polarity['compound'])
#                print('.', end=' ', flush=True)
            else:
                continue
index = np.arange(5)
print('done processing files')
plt.xticks(index, ('1', '2', '3', '4', '5'))
print('xticks worked')
plt.scatter(stars, pscore, alpha=.003, c='b', marker='o')
print('plot made!')
fig = plt.gcf()
print('figure made')
fig.savefig('CovarianceMatrix.png')
