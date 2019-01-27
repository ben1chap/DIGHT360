# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:01:20 2018

@author: Ben
The differences between the frequency distributions is pretty significant.
For example the LY and IN all have comma's as their most frequent token,
whereas the others have a period. Additionally the OP actually has a 
significant number of exclamation points in its freqdist. I plan on using these
freqdists as variables in each of my predictive models to help the machine
predict better which texts are which types of sources.
"""

from glob import glob
from nltk.corpus import stopwords
from nltk import word_tokenize
import re

stop_set = set(stopwords.words('english'))
registers = set()
words = {}
frequencies = {}
files = glob('Mini-CORE\\*.txt')
for fileName in files:
    register = fileName[12:14]
    if register not in registers:
        registers.add(register)
        words[register] = set()
        frequencies[register] = {}
    with open(fileName, 'r') as file:
        rawText = file.read()
    text = re.findall(r'<(?:h|p)>(.*)', rawText)
    for line in text:
        tokens = word_tokenize(line)
        for token in tokens:
            if token.lower() in stop_set:
                continue
            if not token.lower() in words[register]:
                words[register].add(token.lower())
                frequencies[register][token.lower()] = 0
            frequencies[register][token.lower()] += 1
for register in registers:
    with open('Mini-CORE\\' + register + '-frequencies.tsv', 'w') as file:
        data = sorted(frequencies[register].items(),
                      key=lambda x: x[1],
                      reverse=True)
        for word, freq in data:
            print(word, freq, sep='\t', file=file)
