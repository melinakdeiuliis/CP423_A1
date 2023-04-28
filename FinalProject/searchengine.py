"""
Search Engine Project 
CP423 - April 2023
Anastasia, Melina and Ryan
"""

import soundex
s = soundex.Soundex()
from collections import Counter
import math
import os
import requests
import hashlib
inverted_index=None

import requests
from nltk.tokenize import word_tokenize
import pandas as pd
from bs4 import BeautifulSoup
from string import punctuation
import numpy as np
import pandas as pd
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))
punctuation = set(punctuation)

'''
#------------------------------------------------------------------------------------#
Classes and Functions
#------------------------------------------------------------------------------------#
'''

class InvertedIndex:
    def __init__(inverted_index):
        inverted_index.index = {}
        
    def add_document(inverted_index, doc_hash, text):
        for word in text.split():
            soundex_code=s.soundex(word)
            if soundex_code not in inverted_index.index:
                inverted_index.index[soundex_code] = {}
            if doc_hash not in inverted_index.index[soundex_code]:
                inverted_index.index[soundex_code][doc_hash] = 0
            inverted_index.index[soundex_code][doc_hash] += 1
    def cosine_similarity(self, vector1, vector2):
        dot_product = 0
        for term in vector1:
            if term in vector2:
                dot_product += vector1[term] * vector2[term]
        return dot_product / (self.math(vector1) * self.math(vector2))
    def _get_term(self, soundex_code):
        for term, code in soundex.Soundex().items():
            if code == soundex_code:
                return term

 classifier = MultinomialNB()

def clean_data(text):
    tokens = word_tokenize(text)
    clean_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in punctuation and token.isalpha()]
    return clean_tokens

def topicContentAxes(file):

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
    
    return x, y

def classifier_Training(classifier, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)

    joblib.dump(vectorizer, 'vectorizer.model')
    
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

'''    
#------------------------------------------------------------------------------------#
Search Engine
#------------------------------------------------------------------------------------#
'''

def printoptions():
    print("""Select an option:
1- Collect new documents.
2- Index documents.
3- Search for a query.
4- Train ML classifier.
5- Predict a link.
6- Your story!
7- Exit""")
    
    

def searchengine():
    printoptions()
    opt=input("Please select an option by inputting an integer: ")
    opt=int(opt)
    
    while (opt!=7):
        """Collect new documents
    read sources.txt and create hash.txt for each site"""
        if (opt==1):
            print("reading source.txt and populating data folder")
            with open("sources.txt", 'r') as f:
                for line in f:
                    topic,url=line.strip().split(",")
                    topic_folder = os.path.join("Data", topic)
                    response=requests.get(url)
                    content=response.text
                    h = hashlib.sha1(url.encode()).hexdigest()
                    filename = h + '.txt'
                    fn=os.path.join(topic_folder,filename)
                    with open(fn,'w',encoding='utf-8') as output:
                        output.write(content)
        elif(opt==2):
            global inverted_index
            inverted_index = InvertedIndex()
            for topic in os.listdir("Data"):
                for filename in os.listdir(os.path.join("Data",topic)):
                    with open(os.path.join("Data",topic,filename),'r',encoding='utf-8') as f:
                        content=f.read() 
                    hash=os.path.splitext(filename)[0]
                    inverted_index.add_document(content,hash)
            with open("invertedindex.txt", "w",encoding='utf-8') as f:
                f.write("| Term | Soundex | Appearances (DocHash, Frequency) |\n")
                f.write("|---------|----------------------------------|\n")
                for query in inverted_index.index:
                    soundex_code=s.soundex(query)
                    for doc_hash, freq in inverted_index.index[query].items():
                        f.write(f"| {query} | {soundex_code} | ({doc_hash}, {freq}) |\n")
        elif(opt==3):
            query=input("Please enter query: ")
            query_vector = {}
            query_terms = query.split()
            query_length = len(query_terms)
            term_freq = Counter(query_terms)
            for term in term_freq:
                query_vector[term] = term_freq[term] / query_length
            for i in range(len(query_terms)):
                term = query_terms[i]
                soundex_code = s.soundex(term)
                if soundex_code not in inverted_index.index:
                    # Find the most similar term using Soundex
                    similar_terms = []
                    for code in inverted_index.index:
                        if soundex.compare(soundex_code, code) >= 3:
                            similar_terms.extend(inverted_index.index[code])
                    if not similar_terms:
                        query_terms[i] = ""
                    else:
                        term_freq = Counter(similar_terms)
                        query_terms[i] = max(term_freq, key=term_freq.get)
            query_terms = [term for term in query_terms if term != ""]
            doc_set = set()
            for term in query_terms:
                soundex_code = soundex.soundex(term)
                if soundex_code in inverted_index.index:
                    doc_set.update(inverted_index.index[soundex_code])
            documents = list(doc_set)
            doc_vectors = []
            for doc in documents:
                text = inverted_index.get_document_text(doc)
                tokens = inverted_index.tokenize(text)
                term_freq = Counter(tokens)
                doc_length = math.sqrt(sum([tf**2 for tf in term_freq.values()]))
                vector = {}
                for term in term_freq:
                    vector[term] = term_freq[term] / doc_length
                doc_vectors.append(vector)
            query_vector = {}
            query_length = len(query_terms)
            term_freq = Counter(query_terms)
            for term in term_freq:
                query_vector[term] = term_freq[term] / query_length
            sim_scores = {}
            for i in range(len(documents)):
                sim_scores[documents[i]] = inverted_index.cosine_similarity(query_vector, doc_vectors[i])
            results=sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top 3 document results")
            for x in results:
                print(x)
                
        elif(opt==4):
            classifier = MultinomialNB()
            x, y = topicContentAxes('sources.txt')
            data, cm = classifier_Training(classifier, x, y)
            evaluation_Results(data, cm)

        elif(opt==5):
            link=input("Please enter a link: ")
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
            print("Predicted label: ", label_encoder.inverse_transform(y_pred))
            #print("Confidence score: ", confidence)


        elif(opt==6):
            with open('story.txt', 'r') as f:
                print(f.read())
        else:
            print("Wrong input please try again. Must be integer 1-7")
        
        printoptions()
        opt=input("Please select an option by inputting an integer: ")
        opt=int(opt)


searchengine()
