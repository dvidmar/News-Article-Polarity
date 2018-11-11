# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:41:51 2018

@author: Dave
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import zipfile
import requests

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#NLP processing functions
def tokenizeArticles(articles,article_len,dictionary):
    
    #tokenize articles given word dictionary and convert to padded sequences
    word_to_ind = lambda x: dictionary[x] if x in dictionary else 0
   
    seqs = []
    for article in tqdm(articles):
        text = text_to_word_sequence(article)
        seqs.append([word_to_ind(word) for word in text])
       
    seqs_pad = pad_sequences(seqs, maxlen=article_len)
    
    return seqs_pad

def cleanData(articles,labels):
    stop_words = stopwords.words('english')
    raw_articles, cleaned_articles, cleaned_labels = [], [], []
    
    for n,article in enumerate(tqdm(articles)):
        if len(article)>100: #make sure article has sufficient length
            words = word_tokenize(article)
        
            table = str.maketrans('', '', string.punctuation) #get rid of punctuation
            words = [w.lower().translate(table) for w in words if w.isalpha() and w.lower() not in stop_words] #get rid of stopwords
            
            raw_articles.append(article)
            cleaned_articles.append(' '.join(words)) #reconstruct and separate by a space
            cleaned_labels.append(labels[n]) #mark down the corresponding class
            
    return raw_articles,cleaned_articles, cleaned_labels

def loadCSV(path,max_text_len,word_dic):
    #load in CSV of articles and prepare for training
    data = pd.read_csv(path,encoding='latin1')
    articles = data['Article Text'].values
    labels = data['Affiliation (0=conservative)'].values
    
    #clean text if first run
    if 'cleaned' not in path:
        print('Cleaning Text Data...')
        
        raw_articles, articles, labels = cleanData(np.array(data['Article Text'].fillna(' ').tolist()),labels)
        clean_df = pd.DataFrame({'Raw Text':raw_articles,'Article Text':articles,'Affiliation (0=conservative)':labels})
        clean_df.to_csv(path[:-4] + '_cleaned.csv')
        
    #tokenize and convert to sequences
    X = tokenizeArticles(articles,max_text_len,word_dic)
    Y = to_categorical(labels)
    
    return X,Y

def downloadEmbedding(embed_link):
    #stream in embedding file
    response = requests.get(embed_link, stream = True)
    file_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open('WordEmbedding.zip','wb') as f:
        for content in tqdm(response.iter_content(block_size), total=file_size//block_size, unit='KB', unit_scale=True):
            f.write(content)
            
    #extract zip file in place
    with zipfile.ZipFile('WordEmbedding.zip') as zf:
        zf.extractall('WordEmbedding')

    #remove zipped folder
    os.remove('WordEmbedding.zip')

    
def getEmbedding(embed_dim, path = None, n_words = 400000):
    #if you don't have any word embedding in the proper directory, download it
    if path is None:
        path = 'WordEmbedding'
        if not os.path.isdir('WordEmbedding'):
            print('No word embedding file, downloading...')
            downloadEmbedding('http://nlp.stanford.edu/data/glove.6B.zip')
            print('Embedding files downloaded and extracted!')
    
    #get pre-trained word embeddings
    embed_mat = np.zeros([n_words+1,embed_dim])
    
    #put all words and their embeddings in dictionary, adapted from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    embed_dic, word_dic = {}, {}
    with open(os.path.join(path, 'glove.6B.' + str(embed_dim) + 'd.txt'),encoding="utf8") as f:
        for n,line in enumerate(f):
            word = line.split()[0]
            embed_dic[word] = np.asarray(line.split()[1:],dtype='float32')
            word_dic[word] = n+1
    
    #create embedding matrix
    for word,ind in word_dic.items():
        embed_vec = embed_dic.get(word)
        if embed_vec is not None:
            embed_mat[ind,:] = embed_dic.get(word)
    
    return embed_mat,word_dic    
    
    