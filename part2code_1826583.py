#!/usr/bin/env python
# coding: utf-8

# In[11]:


# !pip install numpy
# !pip install pandas
# !pip install nltk
# !pip install sklearn
# !pip install operator
# !pip install requests
import numpy as np
import nltk
import sklearn
import operator
import requests
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed


# In[5]:


import pandas as pd
url_dev_pos='imdb_dev_pos.txt'
url_dev_neg='imdb_dev_neg.txt'

url_train_pos='imdb_train_pos.txt'
url_train_neg='imdb_train_neg.txt'

url_test_pos='imdb_test_pos.txt'
url_test_neg='imdb_test_neg.txt'


dev_pos = open(url_dev_pos).readlines()
train_pos = open(url_train_pos).readlines()
test_pos = open(url_test_pos).readlines()
print("dev-pos-len=", len(dev_pos), "train_pos-len=", len(train_pos),"test_pos-len=", len(test_pos))

dev_neg = open(url_dev_neg).readlines()
train_neg = open(url_train_neg).readlines()
test_neg = open(url_test_neg).readlines()
print("dev-neg-len=", len(dev_neg),"train-neg-len=", len(train_neg),"test-neg-len=", len(test_neg))

# print(pd.DataFrame(dev_pos))


# In[6]:


#def a methｄ to put together positive and negtive data in a single list

def getSingleList(data_pos,data_neg):
    dataset_full=[]
    for pos_review in data_pos:
        dataset_full.append((pos_review,1))
    for neg_review in data_neg:
        dataset_full.append((neg_review,0))
    return dataset_full


# In[7]:


import random
dev_full= getSingleList(dev_pos,dev_neg)
test_full = getSingleList(test_pos,test_neg)
train_full = getSingleList(train_pos, train_neg)
random.shuffle(dev_full)
random.shuffle(test_full)
random.shuffle(train_full)

print ("Size dev full: "+str(len(dev_full)))
print ("Size training set: "+str(len(train_full)))
print ("Size test set: "+str(len(test_full)))
# print(train_full[0])
# print(pd.DataFrame(train_full))


# In[14]:


# import nltk
# nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
stp_set=set(nltk.corpus.stopwords.words('english'))
stp_set.add(".")
stp_set.add(",")
stp_set.add("--")
stp_set.add("``")
stp_set.add("'")
stp_set.add("/")
stp_set.add("\\")
stp_set.add("\"")
stp_set.add("!")
stp_set.add("?")
stp_set.add("...")
stp_set.add("<br />")
stp_set.add("(")
stp_set.add(")")


# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

# Function taken from Session 2
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text


# Functions taken from Session 3

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
    for word in sentence_tokens:
      if word in stp_set: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary

def get_separate_vector_set(training_set,vocabulary):
    X_test=[]
    Y_test=[]
    for instance in training_set:
      vector_instance=get_vector_text(vocabulary,instance[0])
      X_test.append(vector_instance)
      Y_test.append(instance[1])
    X_test=np.asarray(X_test)
    Y_test=np.asarray(Y_test)
    return X_test,Y_test

def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train,Y_train=get_separate_vector_set(training_set,vocabulary)
  
  # Finally, we train the SVM classifier 
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(X_train,Y_train)
  return svm_clf


# In[ ]:


# TF(term frequency)
vocabulary=get_vocabulary(train_full, 1000)  # We use the get_vocabulary function to retrieve the vocabulary
# print("vocabulary:", vocabulary)


# In[ ]:


print("start train svm, This can take a while...")
svm_clf=train_svm_classifier(train_full, vocabulary) # We finally use the function to train our SVM classifier. This can take a while...


# In[ ]:


#evaluate
from sklearn.metrics import classification_report
X_test,Y_test=get_separate_vector_set(test_full, vocabulary)
Y_test_predictions=svm_clf.predict(X_test)
print("TF & SVM")
print(classification_report(Y_test, Y_test_predictions))


# In[75]:


# clean sentence
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
 
def cleanString(sentence):
    # remove HTML、XML
    t1 = BeautifulSoup(sentence, "lxml").get_text()
 
    # remove non-letters
    non_letter = re.sub("[^a-zA-Z]", " ", t1)
 
    # uper to lower
    words = non_letter.lower().split()
 
    # convert stop Words dic
    stops = set(stopwords.words("english"))
 
    # remove stop Words
    meaningful_words = [w for w in words if not w in stp_set]
 
    return " ".join(meaningful_words)
 
 


# In[74]:


#TF-IDF
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
#Instantiate participle objects with default parameters
count_vec=CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)


clean_train = []
train_label=[]
for string in dev_pos:
    clean_train.append(cleanString(string))
    train_label.append(1)
for string in dev_neg:
    clean_train.append(cleanString(string))
    train_label.append(0)
# By calling CountVectorizer's fit_transform method, we built the vocabulary library of the word bag model.
bag=count_vec.fit_transform(clean_train)
print("feature count:",len(count_vec.vocabulary_))
print("bag.shape:", bag.shape)
 
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bag) 

# print (tfidf)


# In[ ]:


#Logistic Regression
from scipy.sparse import bsr_matrix
from sklearn.linear_model import LogisticRegression
 
# bsr matrix
bag = bsr_matrix(bag)
print('bag.shape: ', bag.shape)
 
# trainning data
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None) 
lr.fit(bag, list(train_label))

svm_clf.fit(bag,list(train_label))


# In[ ]:


#Perform the same text processing on the test set, and then input the model for prediction.
clean_test = []
test_label=[]
for string in test_pos:
    clean_test.append(cleanString(string))
    test_label.append(1)
for string in test_neg:
    clean_test.append(cleanString(string))
    test_label.append(0)
   
test_data_features = count_vec.transform(clean_test)
test_data_features = bsr_matrix(test_data_features)
 
result = lr.predict(test_data_features)
print("TF-IDF & LogisticRegression")
print(classification_report(test_label, result))
 


# In[ ]:


#Affective polarity analysis
import sys
import re
import codecs
import os
import jieba
import gensim, logging
from gensim.models import word2vec
import joblib
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc



def parseSent(sentence):
    seg_list = jieba.cut(sentence)
    output = ''.join(list(seg_list)) # use space to join them
    return output
 # Segment a sentence to words
def sent2word(sentence):
    segResult = []


    segList = jieba.cut(sentence)

    for w in segList:
        segResult.append(w)

    newSent=[]
    stopwords_list=[]
    for word in segResult:
        newSent.append(word)

    return newSent

def getWordVecs(wordList):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype = 'float')

def buildVecs2(train_set):
  train=[]
  for line in train_set:            
      line = list(jieba.cut(line))

      resultList = getWordVecs(line)

      # for each sentence, the mean vector of all its vectors is used to represent this sentence
      if len(resultList) != 0:
         resultArray = sum(np.array(resultList))/len(resultList)
         train.append(resultArray)
  return train


# In[ ]:


# load word2vec model
sentences=[]
for x in clean_dev:
    sentences.append(sent2word(x))

model = gensim.models.Word2Vec(sentences, min_count=1)
X_dev3=buildVecs2(clean_dev)
X_test3=buildVecs2(clean_test)


# In[ ]:


# SVM (RBF)
clf = SVC(C = 2, probability = True)
clf.fit(X_dev3, dev_label)
print ("Test Accuracy: %.2f"%clf.score(X_dev3, dev_label))

pred_probas = clf.predict_proba(X_test3)[:,1]

test_value=clf.predict(X_test3)

#output results
print("Affective polarity analysis & SVM")
print(classification_report(test_label, test_value))


# In[ ]:




