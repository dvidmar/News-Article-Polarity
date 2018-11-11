# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:07:52 2018

@author: Dave
"""
import numpy as np
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
        
class CNN():
    
    def __init__(self,embed_mat,max_article_len):
        self.embed_mat = embed_mat
        self.n_words = np.shape(self.embed_mat)[0]
        self.embed_dim = np.shape(self.embed_mat)[1]
        self.max_article_len = max_article_len
        
        
        #adapted from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        embedding_layer = Embedding(self.n_words,
                                    self.embed_dim,
                                    weights=[self.embed_mat],
                                    input_length=self.max_article_len,
                                    trainable=False)
        
        # train a 1D convnet with global maxpooling
        self.sequence_input = Input(shape=(self.max_article_len,), dtype='int32')
        embedded_sequences = embedding_layer(self.sequence_input)
        x = SpatialDropout1D(0.5)(embedded_sequences)
        
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        self.preds = Dense(2, activation='softmax')(x)
            