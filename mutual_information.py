#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:57:32 2018

@author: david
"""
import numpy as np

#Mutual Information Functions
def class_table(word,class_token,class_article_len,not_class_token,not_class_article_len):
    arr = np.zeros([2,2])
    
    if word in class_token.word_index:
        n_class_docs = class_token.word_docs[word]
    else:
        n_class_docs = 0
        
    if word in not_class_token.word_index:
        n_not_class_docs = not_class_token.word_docs[word]
    else:
        n_not_class_docs = 0
    
    arr[0,0] = n_class_docs
    arr[1,0] = class_article_len-n_class_docs
    arr[0,1] = n_not_class_docs
    arr[1,1] = not_class_article_len-n_not_class_docs
    return arr

def mutual_info(arr_raw):
    arr = np.copy(arr_raw)+1e-10
    
    if arr[0,0]/(arr[1,0]+arr[0,0])>arr[0,1]/(arr[1,1]+arr[0,1]):
        #member of class if word is present, return MI
        return (arr[0,0]/arr.sum())*np.log2(arr.sum()*arr[0,0]/((arr[0,0]+arr[0,1])*(arr[0,0]+arr[1,0])) ) + \
        (arr[1,0]/arr.sum())*np.log2(arr.sum()*arr[1,0]/((arr[1,0]+arr[0,0])*(arr[1,0]+arr[1,1])) ) + \
        (arr[0,1]/arr.sum())*np.log2(arr.sum()*arr[0,1]/((arr[0,1]+arr[0,0])*(arr[0,1]+arr[1,1])) ) + \
        (arr[1,1]/arr.sum())*np.log2(arr.sum()*arr[1,1]/((arr[1,0]+arr[1,1])*(arr[1,1]+arr[0,1])) )
    else:
        #member of class if word is absent, not interesting for us
        return 0.0
