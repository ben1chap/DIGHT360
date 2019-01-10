# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:18:56 2018

@author: Ben
"""
import json
import requests as r
from random import randint
from time import sleep

from yelpapi import YelpAPI

api_key = ('qzEmtOATJNAuUIx68h2qzU4d5oHkBVIv53M505outf4Q2dO2DfBme \
           _zq1cn5opzCjqHylv3AK1hrvjTibJ4taXgS_3CwGDl4IFAJ6XvWs7p\
           i5xg1Akf8R-hOd1AAXHYx')


yelp_api = YelpAPI(api_key, timeout_s=3.0)
genres = ['american', 'italian', 'chinese', 'mexican', 'japanese', 'indian',
          'spanish', 'greek', 'french', 'british', 'korean', 'caribbean',
          'thai', 'vietnamese', 'vegan', 'gluten-free', 'cajun', 'filipino',
          'jewish', 'mediterranean', 'fast']

for genre in genres:
    total = 20000
    offset = 0
    while offset < total:
        offset = offset + 50
        data = yelp_api.search_query(term=genre, location='utah', offset=offset)
        total = search_results['total']
        with open('slugs', 'a') as file:
            for business in data['businesses']:
                alias = business['alias']
                name = business['name']
                print(alias, file=file)
    print(alias)
with open('slugs') as s:
    slugs = set(line.strip() for line in s)
