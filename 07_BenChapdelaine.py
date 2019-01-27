# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:03:47 2018

@author: Ben
"""


"""Assignment 7 Ben Chapdelaine"""

import glob
import re
from string import punctuation as punct  # string of common punctuation chars
from math import log1p #fixes log(0) error
from statistics import mean

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import model classes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO change to the location of your Mini-CORE corpus
MC_DIR = 'Mini-CORE\\'
STOPWORDS = set(stopwords.words('english'))
INWL = set(['content', 'explanation', 'firstly', 'lastly', 'furthermore', 'therefore', 'fact', 'also', 'year', 'place', 'locate', 'located', 'location'])
IPWL = set(['conclude', 'argument', 'thesis', 'reason', 'explanation', 'firstly',
                'lastly', 'furthermore', 'therefore', 'analyze', 'analysis',
                'always', 'never', 'best', 'certain', 'confident',
                'definitely', 'expect', 'superior', 'inferior', 'shameful',
                'unreliable', 'biased', 'bias', 'cruel', 'aggravate',
                'confusing'])
LYWL = set(['love', 'sad', 'feel', 'sorry', 'apologize', 'happy', 'mad', 'baby', 'good', 'bad', 'fall', 'heart', 'body', 'dance', 'alone', 'lonely', 'like', 'interested', 'yeah', 'ya', 'hate', 'leave', 'lost', 'empty', 'think', 'time', 'free', 'beautiful', 'babe', 'la', 'oh', 'o'])
NAWL = set(['Fury', 'Rage', 'Outrage', 'Mull', 'Romp', 'Rumpus', 'Sex', 'Lauded', 'Slay', 'Flap', 'Nab', 'Vie', 'Row', 'Ire', 'Inks', 'Slammed', 'Tot', 'Vowed', 'Quizzed', 'Wed', 'Roared’, ‘Tired', 'emotional', 'Blasted', 'Poised', 'Tout', 'Cash', 'Back', 'Scare', 'Probe', 'Woe', 'Spar', 'Bid', 'Aims', 'to Eye', 'Urge', 'Linked', 'Weighs', 'Unveil', 'Draws', 'Ready', 'Scrutiny', 'hits', 'tops', 'climbs', 'dips', 'lags', 'surges', 'spikes', 'tumbles', 'flops', 'Assets', 'Charms', 'Manhood', 'Roly-poly', 'Funnyman', 'Dubbed', 'Alleged', 'Suspect', 'Jibe', 'Embattled', 'Gunman', 'Extremist', 'Moderate'])
OPWL = set(['unacceptable', 'believe', 'think', 'feel', 'opinion', 'favorite', 'best', 'worst', 'view'])
SPWL = set(['?', 'experience', 'skills', 'skill', 'training', 'plan',
                'vision', 'company', 'hire', 'strengths', 'weakness', 'job',
                'profession', 'career', 'challenge', 'conflict', 'work',
                'achievement', 'interview', 'leadership', 'management',
                'decision', 'pressure', 'stress', 'culture', 'resume',
                'success', 'failure', 'hobbies', 'industry' ])



def clean(in_file):
    """Remove headers from corpus file."""
    out_str = ''
    for line in in_file:
        if re.match(r'<[hp]>', line):
            out_str += re.sub(r'<[hp]>', '', line)
    return out_str


def subcorp(name):
    """Extract subcorpus from filename.

    name -- filename

    The subcorpus is the first abbreviation after `1+`.
    """
    return name.split('+')[1]


def ttr(in_Text):
    """Compute type-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    return len(set(in_Text)) / len(in_Text)


def pro1_tr(in_Text):
    """Compute 1st person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:i|me|my|mine)$'
    pro1_count = len([i for i in in_Text if re.match(regex, i, re.I)])
    return pro1_count / len(in_Text)


def pro2_tr(in_Text):
    """Compute 2nd person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:ye|you(?:rs?)?)$'
    pro2_count = len([i for i in in_Text if re.match(regex, i, re.I)])
    return pro2_count / len(in_Text)


def pro3_tr(in_Text):
    """Compute 3rd person pronoun-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    regex = r'(?:he|him|his|she|hers?|its?|they|them|theirs?)$'
    pro3_count = len([i for i in in_Text if re.match(regex, i, re.I)])
    return pro3_count / len(in_Text)


def punct_tr(in_Text):
    """Compute punctuation-token ratio for input Text.

    in_Text -- nltk.Text object or list of strings
    """
    punct_count = len([i for i in in_Text if re.match('[' + punct + ']+$', i)])
    return punct_count / len(in_Text)


def prop_of_N(tagged):
    return len([tok for tok, tag in tagged if tag.startswith('N')])/len(tagged)


def paragraphcount(text): # does clean text look slike this
   para = text.count('<p>')
   tokens = len(raw_text)
   return tokens / (para + 1)


def pnr(text):
   """(count # of I, me, my, mine)/(count # of selfish + count of nonselfish)"""
   selfish = r'\b(I|me|(my.*?)|mine)\b'
   others = r'\b(you|you.*?|ye)\b'
   selfcount = re.findall(selfish, raw_text, re.I)
   otherscount = re.findall(others, raw_text, re.I)
   try:
       return len(selfcount) / (len(selfcount) + len(otherscount))
   except ZeroDivisionError:
       return 0


def fd(fdist, targetset):
    return mean([log1p(fdist[word]) for word in targetset])


# TODO add feature names HERE
feat_names = ['ttr', '1st-pro', '2nd-pro', '3rd-pro', 'punct', 'noun-ratio',
    'tok/para', 'selfish-ratio', 'INFreqDist', 'IPFreqDist', 'NAFreqDist',
    'LYFreqDist', 'OPFreqDist', 'SPFreqDist', 'genre']# addd stufff herrrrreeee

with open('mc_feat_names.txt', 'w') as name_file:
    name_file.write('\t'.join(feat_names))


with open('mc_features.csv', 'w') as out_file:
    for f in glob.glob(MC_DIR + '*.txt'):
        print('.', end='', flush=True)  # show progress; print 1 dot per file
        with open(f) as the_file:
            raw_text = clean(the_file)
            tok_text = nltk.word_tokenize(raw_text)
            tagged = nltk.pos_tag(tok_text)
            fdist = FreqDist([token for token in tok_text if token not in STOPWORDS])
        # TODO call the function HERE
        print(ttr(tok_text), pro1_tr(tok_text), pro2_tr(tok_text),
              pro3_tr(tok_text), punct_tr(tok_text), prop_of_N(tagged),
              paragraphcount(raw_text), pnr(raw_text), fd(fdist, INWL),
              fd(fdist, IPWL), fd(fdist, NAWL), fd(fdist, LYWL),
              fd(fdist, OPWL), fd(fdist, SPWL), subcorp(f),sep=',', file=out_file)
    print()  # newline after progress dots

##############################################################################
with open('mc_feat_names.txt') as name_file:
    names = name_file.read().strip().split('\t')
len_names = len(names)
with open('mc_features.csv') as mc_file:
    dataset = pandas.read_csv(mc_file, names=names,  # pandas DataFrame object
                              keep_default_na=False, na_values=['_'])  # avoid 'NA' category being interpreted as missing data  # noqa
print(type(dataset))

# Summarize the data
print('"Shape" of dataset:', dataset.shape,
      '({} instances of {} attributes)'.format(*dataset.shape))
print()
print('"head" of data:\n', dataset.head(20))  # head() is a method of DataFrame
print()
print('Description of data:\n:', dataset.describe())
print()
print('Class distribution:\n', dataset.groupby('genre').size())
print()

# Visualize the data
print('Drawing boxplot...')
grid_size = 0
while grid_size ** 2 < len_names:
    grid_size += 1
dataset.plot(kind='box', subplots=True, layout=(grid_size, grid_size),
             sharex=False, sharey=False)
fig = plt.gcf()  # get current figure
fig.savefig('boxplots.png')

# histograms
print('Drawing histograms...')
dataset.hist()
fig = plt.gcf()
fig.savefig('histograms.png')

# scatter plot matrix
print('Drawing scatterplot matrix...')
scatter_matrix(dataset)
fig = plt.gcf()
fig.savefig('scatter_matrix.png')
print()

print('Splitting training/development set and validation set...')
# Split-out validation dataset
array = dataset.values  # numpy array
feats = array[:,0:len_names - 1]  # to understand comma, see url in next line:
labels = array[:,-1]  # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
print('\tfull original data ([:5]) and their respective labels:')
print(feats[:5], labels[:5], sep='\n\n', end='\n\n\n')
validation_size = 0.20
seed = 7  # used to make 'random' choices the same in each run
split = model_selection.train_test_split(feats, labels,
                                         test_size=validation_size,
                                         random_state=seed)
feats_train, feats_validation, labels_train, labels_validation = split
# print('\ttraining data:\n', feats_train[:5],
#       '\ttraining labels:\n', labels_train[:5],
#       '\tvalidation data:\n', feats_validation[:5],
#       '\tvalidation labels:\n', labels_validation[:5], sep='\n\n')

# Test options and evaluation metric
print()

print('Initializing models...')
# Spot Check Algorithms
models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC())]
print('Training and testing each model using 10-fold cross-validation...')
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # https://chrisjmccormick.files.wordpress.com/2013/07/10_fold_cv.png
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, feats_train,
                                                 labels_train, cv=kfold,
                                                 scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = '{}: {} ({})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
print()

print('Drawing algorithm comparison boxplots...')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig = plt.gcf()
fig.savefig('compare_algorithms.png')
print()

# Make predictions on validation dataset
# best_model = KNeighborsClassifier()
# best_model.fit(feats_train, labels_train)
# predictions = best_model.predict(feats_validation)
# print('Accuracy:', accuracy_score(labels_validation, predictions))
# print()
# print('Confusion matrix:')
# cm_labels = 'Iris-setosa Iris-versicolor Iris-virginica'.split()
# print('labels:', cm_labels)
# print(confusion_matrix(labels_validation, predictions, labels=cm_labels))
# print()
# print('Classification report:')
# print(classification_report(labels_validation, predictions))'''




