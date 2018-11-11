# -*- coding: utf-8 -*-
"""
Scrape N articles at random from a liberal subreddit and a conservative subreddit over a given time interval, 
and save the resulting article texts and class labels in a CSV file.

Created on Tue Apr 17 18:14:48 2018

@author: Dave
"""
import argparse        
import newspaper
import datetime
import pandas as pd
import tqdm

from scraper_utils import getPosts, NewsSource

# Get CLI Arguments -----------------------------------------------------------
parser = argparse.ArgumentParser(description='Select Parameters for Article Scraping.')

parser.add_argument('--batch_size', default = 100, help='Batch size for article scraping operation (DEFAULT 100).')
parser.add_argument('--n_articles', default = 50000, help='Number of articles to get (DEFAULT 50000).')
parser.add_argument('--start_year', default = 2015, help='Oldest year to scrape from (DEFAULT 2015).')
parser.add_argument('--start_month', default = 1, help='Oldest year to scrape from (DEFAULT 2015).')
parser.add_argument('--start_day', default = 1, help='Oldest year to scrape from (DEFAULT 2015).')
parser.add_argument('--cons_reddit', default = 'Conservative', help='Conservative subreddit (DEFAULT /r/Conservative).')
parser.add_argument('--lib_reddit', default = 'politics', help='Liberal subreddit (DEFAULT /r/politics).')
parser.add_argument('--save_name', default = 'Articles.csv', help='Save name for articles (DEFAULT "Articles.csv").')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
N_ARTICLES = int(args.n_articles)
START_TIME = datetime.date(int(args.start_year),int(args.start_month),int(args.start_day))
CONS_REDDIT = args.cons_reddit
LIB_REDDIT = args.lib_reddit
SAVE_NAME = args.save_name
    
# Scrape and Save Articles ----------------------------------------------------

#scrape articles from random times in the interval (start_time,present)
articles = [' ']*N_ARTICLES
all_labels = [[] for i in range(N_ARTICLES)]
all_links = [[] for i in range(N_ARTICLES)]
all_domains = [[] for i in range(N_ARTICLES)]

for i in tqdm.tqdm(range(int(N_ARTICLES/BATCH_SIZE))):
    try:
        #get urls of random posts
        urls, article_labels, domains, full_links = getPosts(START_TIME,CONS_REDDIT,LIB_REDDIT,BATCH_SIZE)
        
        #scrape urls with multi-threading
        sources = [NewsSource(url=url) for url in urls]
        newspaper.news_pool.set(sources)
        newspaper.news_pool.join()
        
        #add these labels and domains to the full vector
        all_labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = article_labels
        all_links[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = full_links
        all_domains[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = domains
        
        #parse each scraped url and save article text
        for n,source in enumerate(sources):
            article = source.articles
            if len(article)>0:
                try: #try to parse
                    article[0].parse()
                    articles[i*BATCH_SIZE+n] = article[0].text
                except: #if it can't be parsed, move on
                    pass
    except:
        pass

#save results in a DataFrame
df = {}
df['Article Text'] = articles
df['Affiliation (0=conservative)'] = all_labels
df['Full Link'] = all_links
df['Website'] = all_domains
df = pd.DataFrame(df)

#drop all unscrapable batches and save as csv
df=df.drop([i for i,x in enumerate(all_labels) if x == []]).reset_index(drop=True)
if 'csv' in SAVE_NAME:
    df.to_csv(SAVE_NAME)
else:
    df.to_csv(SAVE_NAME+'.csv')

