# Import libraries
import argparse
import json
import numpy as np
import string
import os
import nltk
import pandas as pd
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifier libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

# Performance of model and confusion matrix libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Tokenize and stopwords libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))
punctuation = set(punctuation)

# Dataset file names
filepath_dict = {'--imdb': 'imdb_labelled.txt',
                '--yelp': 'yelp_labelled.txt',
                '--amazon': 'amazon_cells_labbeled'}

# Testing dataset
fileName = filepath_dict['--imdb']

# Data from file
data_list = []

# Dataset seperated by 'text' and 'sentiment'
dataset = pd.read_csv(fileName, names = ['text', 'sentiment'], sep = '\t')

# Tokenize and remove stop words and punctuation
def clean_data(text):
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in punctuation and token.isalpha()]
    return clean_tokens

# Replace original text with clean_data
dataset['text'] = dataset['text'].apply(clean_data)

data_list.append(dataset)
dataset = pd.concat(data_list)

#print(dataset)

# for entry in dataset['text']:
#     vectorizer = CountVectorizer(min_df=0, lowercase=True)
#     vectorizer.fit(entry)
#     print(vectorizer.vocabulary_)


records = dataset['text'].values
# x = dataset['text'].values
y = dataset['sentiment'].values


# Sorts the records by list of strings
x = [' ' .join(doc) for doc in records]

# print(x)

# vectorizer = CountVectorizer(tokenizer = lambda x: x, preprocessor=lambda x: x)

# Corpus
vectorize = CountVectorizer()
corpus = vectorize.fit_transform(x)

# Prints the sentimate values for each word in list of strings
#print(corpus.toarray())

#print(corpus)

# vectorizer = CountVectorizer(vocabulary=corpus)

#print(x)

# Prints the index of each unique word in corpus
#print("Vocabulary: ", vectorizer.vocabulary_)

#print(x.toarray())

#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

vectorizer = CountVectorizer()
#vectorizer.vocabulary_

vectorizer.fit(x_train)
X_training = vectorizer.transform(x_train)
X_testing = vectorizer.transform(x_test)

classifier = MultinomialNB()
classifier.fit(X_training, y_train)

# Evaluate the classifier's performance on the testing set
y_pred = classifier.predict(X_testing)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix = matrix)
display.plot()
plt.show()
