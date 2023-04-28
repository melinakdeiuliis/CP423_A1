import requests
from nltk.tokenize import word_tokenize
import pandas as pd
from bs4 import BeautifulSoup
from string import punctuation

# Import libraries
import argparse
import json
import numpy as np
import string
import os
import nltk
import pandas as pd
from string import punctuation
import pickle

# Classifier libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

# Performance of model and confusion matrix libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Tokenize and stopwords libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))
punctuation = set(punctuation)

def clean_data(text):
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in punctuation and token.isalpha()]
    return clean_tokens

link = 'http://sten.astronomycafe.net/2023/03/'

response = requests.get(link)
content = response.text
soup = BeautifulSoup(content, 'html.parser')

text = soup.get_text()

text_list = []

text = clean_data(text)

text_list.append(text)

x = [' ' .join(text)]

classifier = joblib.load('classifier.model')
vectorizer = joblib.load('vectorizer.model')

X_testing = vectorizer.transform(x)

label_encoder = LabelEncoder()
label_encoder.classes_ = classifier.classes_

y_pred = classifier.predict(X_testing)

#confidence = classifier.predict_proba(X_testing)[0][y_pred]
print(y_pred)
# Print the predicted label and its confidence score
print("Predicted label:", label_encoder.inverse_transform(y_pred))
#print("Confidence score:", confidence)
