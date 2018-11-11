#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:52:30 2018

@author: david
"""
import time, datetime
import newspaper
import json, requests
from sklearn.utils import shuffle
import numpy as np

#Article Scraping Functions 
class NewsSource(newspaper.Source):
    #news source class for multi-threading purposes
    def __init__(self, url):
        super(NewsSource, self).__init__("http://localhost")
        self.articles = [newspaper.Article(url=url)]

def getURL(t,batch_size,subreddit):
    #get pushshift url of latest N articles starting from t
    return 'https://api.pushshift.io/reddit/search/submission?&size=' + str(batch_size) \
                                                + '&before=' + str(t)+'&subreddit='+str(subreddit) 
                                                
def getRandomTime(start_date):
    #random timestamp from (start_date,present)
    now_time = int(time.mktime(datetime.datetime.now().timetuple()))
    start_time = int(time.mktime(start_date.timetuple()))
    
    return np.random.randint(start_time,now_time)

def getPosts(start_date,cons_reddit,lib_reddit,batch_size):
    #get post urls and labels from a random time period (both liberal and conservative)
    rand_time = getRandomTime(start_date)
    posts = json.loads(requests.get(getURL(rand_time,batch_size,cons_reddit)).text)['data'] + \
            json.loads(requests.get(getURL(rand_time,batch_size,lib_reddit)).text)['data']
    labels = [0]*batch_size + [1]*batch_size
    
    #shuffle posts
    rand_seed = np.random.randint(1e8)
    posts = shuffle(posts, random_state = rand_seed)
    labels = shuffle(labels, random_state = rand_seed)
    
    #get article urls from these posts
    links, article_labels,domains,full_links = [], [], [], []
    n = 0
    
    while len(links)<batch_size and n<int(2*batch_size):
        link = posts[n]['url']
        full_link = posts[n]['full_link']
        domain = posts[n]['domain']
        label = labels[n]
        n+=1
    
        if 'reddit.com/' not in link and 'google.com/' not in link and \
            'youtube.com/' not in link:
            links.append(link)
            full_links.append(full_link)
            domains.append(domain)
            article_labels.append(label)
    
    return links, article_labels,domains,full_links