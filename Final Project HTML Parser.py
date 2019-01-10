# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:34:36 2018
Final Project

@author: Ben
"""
from time import sleep
from pathlib import Path
import re
import requests as r
import random
import os.path

from bs4 import BeautifulSoup


def get_soup(newurl):
    """Scrape html file from yelp.com, return soup"""
    sleep(random.uniform(1.5, 2.5))
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110'
         'Safari/537.36'}
    req = r.get(newurl, headers=h)
    soup = BeautifulSoup(req.text, 'html5lib')
    return soup


def get_rev(soup):
    """return a numeric review and text content of review; save to file."""

    reviews = []
    for review in soup.find_all('div', class_='review-wrapper'):
        try:
            star = review.find('div', class_='i-stars')['title'][0]
            review = review.p.get_text()
            reviews.append((star, review))
        except TypeError:
            continue
    return reviews


def there_is_more(soup):
    """return url if there are more reviews to scrape"""

    found_tag = soup.find('a', class_='next')
    if found_tag:
        return found_tag.get('href')


reviews = []

with open('slugs') as s:
    slugs = set(line.strip() for line in s)

for slug in slugs:
    if os.path.exists(
            f'C:\\Users\\Ben\\Desktop\\F18_DIGHT360\\final-corpus\\{slug}'):
        print('skip')
        continue
    newurl = f'https://www.yelp.com/biz/{slug}'
    reviews = []
    print(f'processing {slug}')
    while newurl is not None:
        soup = get_soup(newurl)
        newurl = there_is_more(soup)
        reviews.extend(get_rev(soup))
    with open(f'C:\\Users\\Ben\\Desktop\\F18_DIGHT360\\final-corpus\\{slug}',
              'w', encoding='utf8') as rvw:
        for review in reviews:
            print('\t'.join(review), file=rvw)
