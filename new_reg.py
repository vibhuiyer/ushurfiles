#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:38:40 2019

@author: vibhuiyer
"""

import os
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read.csv("OpenResponseRating.csv")

processed = data['response dispStr'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' ')

stop_words = set(stopwords.words('at', 'and', 'was', 'if'))
processed = processed.apply(lambda x: ' '.join(term for term
                                in x.split() if term not in stop_words))

all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(10)))

word_features = list(all_words.keys())[:1000]

def find_features(message):
    word = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
        
    return features

messages = list(zip(processed, data['Number of Records']))
print('Testing:', messages[0])

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

featuresets = [(find_features(text), label) for (text, label) in messages]
print(featuresets[0])

training, testing = model_selection.train_test_split(featuresets, 
                                                     test_size = 0.2, random_state = 1)

print("training set: ", len(training))
print("testing set: ", len(testing))

model = MultinomialNB()

nltk_model = SklearnClassifier(model)
nltk_model.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Naive Bayes Accuracy: {}".format(accuracy))

txt_features, labels = list(zip(*testing))
prediction = nltk_model.classify_many(txt_features)
print("Mean absolute error:", mean_absolute_error(prediction, labels)*100)

neg_1=[], neg_2=[], neg_3=[], neg_4=[], neg_5=[], neg_6=[]
pos_7=[], pos_8=[], pos_9=[], pos_10=[]
unifyy = list(zip(prediction, txt_features, labels, data['Number of Records']))
for p,t,l,r in unifyy:
    for key, value in t.items():
        if value==True and l==p==0 and r==1:
            neg_1.append(key)
            break
        elif value==True and l==p==0 and r==2:
            neg_2.append(key)
            break
        elif value==True and l==p==0 and r==3:
            neg_3.append(key)
            break
        elif value==True and l==p==0 and r==4:
            neg_4.append(key)
            break
        elif value==True and l==p==0 and r==5:
            neg_5.append(key)
            break
        elif value==True and l==p==0 and r==6:
            neg_6.append(key)
            break
        elif value==True and l==p==1 and r==7:
            pos_7.append(key)
            break
        elif value==True and l==p==1 and r==8:
            pos_8.append(key)
            break
        elif value==True and l==p==1 and r==9:
            pos_9.append(key)
            break
        elif value==True and l==p==1 and r==10:
            pos_10.append(key)
            break
print("Negative reviews with 1 rating", neg_1)
print("Negative reviews with 2 rating", neg_2)
print("Negative reviews with 3 rating", neg_3)
print("Negative reviews with 4 rating", neg_4)
print("Negative reviews with 5 rating", neg_5)
print("Negative reviews with 6 rating", neg_6)
print("Positive reviews with 7 rating", pos_7)
print("Positive reviews with 8 rating", pos_8)
print("Positive reviews with 9 rating", pos_9)
print("Positive reviews with 10 rating", pos_10)








