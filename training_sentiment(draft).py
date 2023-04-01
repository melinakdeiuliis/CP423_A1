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
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

# Performance of model and confusion matrix libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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

def naiveBayes(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

    vectorizer = CountVectorizer()
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

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def plot_Confusion_Matrix(y_test, y_pred):
    
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix = matrix)
    display.plot()
    plt.show()

    return

def knn_Classifier(x, y, k):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    
    X_training = vectorizer.transform(x_train)
    X_testing = vectorizer.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_training, y_train)
    y_pred = knn.predict(X_testing)

    x = vectorizer.fit_transform(x)

    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='weighted')
}
    
    evaluation = cross_validate(knn, x, y, scoring=scoring)

    accuracy = evaluation['test_accuracy'].mean()
    recall = evaluation['test_recall'].mean()
    precision = evaluation['test_precision'].mean()
    f1 = evaluation['test_f1_score'].mean()

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

if __name__ == '__main__':
    # Terminal call
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')

    # Dataset options
    dataset_group = parser.add_argument_group('Dataset options:')
    dataset_group.add_argument('--imdb', action='store_true', help='Use IMDB dataset for training')
    dataset_group.add_argument('--yelp', action='store_true', help = 'Use Yelp dataset for training')
    dataset_group.add_argument('--amazon', action='store_true', help='Use the Amazon dataset for training')

    # Classifier options
    classifier_group = parser.add_argument_group('Classifier options:')
    classifier_group.add_argument('--naive', action='store_true', help='Choose Naive Bayes classifier')
    classifier_group.add_argument('--knn', type=int, help='Choose KNN classifier with K neighbors')
    classifier_group.add_argument('--svm', action='store_true', help='Choose SVM classifier')
    classifier_group.add_argument('--decisiontree', action='store_true', help='Choose Decision Tree classifier')

    args = parser.parse_args()

    if args.imdb:
        datasetName = '--imdb'
    elif args.yelp:
        datasetName = '--yelp'
    elif args.amazon:
        datasetName = '--amazon'

    x, y = textSentimentAxes(datasetName)

    if args.naive:
        naiveBayes(x, y)
    elif args.knn:
        k = args.knn
        knn_Classifier(x, y, k)
    elif args.svm:
        classifier = SVC()
    elif args.decisiontree:
        classifier = DecisionTreeClassifier()
