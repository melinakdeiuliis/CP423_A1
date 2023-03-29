# Import libraries
import argparse
import json
import numpy as np
import string
import os
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Classifier libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

# Tokenize and stopwords libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))

# Dataset file names
filepath_dict = {'imdb': 'imdb_labelled.txt',
                'yelp': 'yelp_labelled.txt',
                'amazon': 'amazon_cells_labbeled'}

# Testing dataset
fileName = filepath_dict['imdb']

# Data from file
data_list = []

# Dataset seperated by 'text' and 'sentiment'
dataset = pd.read_csv(fileName, names = ['text', 'sentiment'], sep = '\t')

# Tokenize and remove stop words and punctuation
def clean_data(text):
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]
    return clean_tokens

# Replace original text with clean_data
dataset['text'] = dataset['text'].apply(clean_data)

data_list.append(dataset)
dataset = pd.concat(data_list)

print(dataset)
