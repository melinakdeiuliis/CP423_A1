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

def topicContentAxes(file):
    
    with open(file, 'r') as file:
        for line in file:
            dataset = pd.read_csv(file, names = ['topic', 'text'], sep = ',')

        for link in dataset['text']:
            response = requests.get(link)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')

            text = soup.get_text()

data_list = []

dataset = pd.read_csv('sources.txt', names = ['topic', 'text'], sep = ',')

i = 0
for link in dataset['text']:

    response = requests.get(link)
    content = response.text

    soup = BeautifulSoup(content, 'html.parser')
    
    dataset['text'][i] = soup.get_text()
    i += 1

dataset['text'] = dataset['text'].apply(clean_data)
#print(dataset['text'])

data_list.append(dataset)
dataset = pd.concat(data_list)

websites = dataset['text'].values
# Sorts the records by list of strings
x = [' ' .join(doc) for doc in websites]
y = dataset['topic'].values

classifier = MultinomialNB()

def classifier_Training(classifier, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    
    X_training = vectorizer.transform(x_train)
    X_testing = vectorizer.transform(x_test)

    classifier.fit(X_training, y_train)

    y_pred = classifier.predict(X_testing)

    x = vectorizer.fit_transform(x)

    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='weighted')
    }
    
    evaluation = cross_validate(classifier, x, y, scoring=scoring)

    accuracy = evaluation['test_accuracy'].mean()
    recall = evaluation['test_recall'].mean()
    precision = evaluation['test_precision'].mean()
    f1 = evaluation['test_f1_score'].mean()

    #classifier.fit(X_training, y_train)
    y_pred = classifier.predict(X_testing)

    accuracy_test = accuracy_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred, average='weighted')
    precision_test = precision_score(y_test, y_pred, average='weighted')
    f1_test = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    data = {
        "Cross-validation results": {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1-score": f1
        },
        "Test results": {
            "Accuracy": accuracy_test,
            "Recall": recall_test,
            "Precision": precision_test,
            "F1-score": f1_test
        }
    }

    # Save classifier model for predict_sentiment.py
    joblib.dump(classifier, 'classifier.model')

    return data, cm

def evaluation_Results(data, cm):

    col_width = 18

    for title, values, in data.items():
        print(f"{title.upper():^{col_width * 2}}")
        print("-" * col_width * 2)
        
        # Print the section content
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"{k:<{col_width}}{v:>{col_width}}")
        elif isinstance(values, list):
            for row in values:
                print("".join([f"{str(item):<{col_width}}" for item in row]))
        print()

    display_Confusion_Matrix(cm)

def display_Confusion_Matrix(matrix):
    display = ConfusionMatrixDisplay(confusion_matrix = matrix)
    display.plot()
    plt.show()

data, cm = classifier_Training(classifier, x, y)

evaluation_Results(data, cm)