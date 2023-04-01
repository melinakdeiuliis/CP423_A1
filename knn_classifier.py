# Import libraries
import argparse
import json
import numpy as np
import string
import os
import nltk
import pandas as pd
from string import punctuation

# Classifier libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from nltk.stem import WordNetLemmatizer

# Performance of model and confusion matrix libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Tokenize and stopwords libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))
punctuation = set(punctuation)

def clean_data(text):
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in punctuation and token.isalpha()]
    return clean_tokens

def textSentimentAxes(file):

    filepath_dict = {'--imdb': 'imdb_labelled.txt',
                    '--yelp': 'yelp_labelled.txt',
                    '--amazon': 'amazon_cells_labelled.txt'}

    # Testing dataset
    fileName = filepath_dict[file]

    # Data from file
    data_list = []

    # Dataset seperated by 'text' and 'sentiment'
    dataset = pd.read_csv(fileName, names = ['text', 'sentiment'], sep = '\t')

    # Replace original text with clean_data
    dataset['text'] = dataset['text'].apply(clean_data)

    data_list.append(dataset)
    dataset = pd.concat(data_list)

    records = dataset['text'].values
    # Sorts the records by list of strings
    x = [' ' .join(doc) for doc in records]
    y = dataset['sentiment'].values

    # Corpus
    vectorize = CountVectorizer()
    corpus = vectorize.fit_transform(x)

    # Prints the index of each unique word in corpus
    #print("Vocabulary: ", vectorizer.vocabulary_)

    return x, y

fileName = '--imdb'

vectorizer = CountVectorizer()

x, y = textSentimentAxes(fileName)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

vectorizer.fit(x_train)

X_training = vectorizer.transform(x_train)
X_testing = vectorizer.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_training, y_train)
y_pred = knn.predict(X_testing)

evaluation = cross_val_score(knn, x, y, cv=5)

accuracy = evaluation['test_accuracy'].mean()
recall = evaluation['test_recall_macro'].mean()
precision = evaluation['test_precision_macro'].mean()
f1 = evaluation['test_f1_macro'].mean()

knn.fit(X_training, y_train)

y_pred = knn.predict(X_testing)

accuracy_test = accuracy_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred, average='macro')
precision_test = precision_score(y_test, y_pred, average='macro')
f1_test = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("Cross-validation results:")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("Test results:")
print("Accuracy:", accuracy_test)
print("Recall:", recall_test)
print("Precision:", precision_test)
print("F1-score:", f1_test)
print("Confusion matrix:\n", cm)
