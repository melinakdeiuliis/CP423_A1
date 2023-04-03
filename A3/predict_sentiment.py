import pickle
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from training_sentiment import clean_data,
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load classifier model from training_sentiment.py
with open('classifier.pkl', 'rb') as classifier:
    model = pickle.load(classifier)

with open('vectorizer.pkl', 'rb') as vector:
    vectorizer = pickle.load(vector)



def predict_sentiment(string):

    #vectorizer = CountVectorizer

    # Tokenize and clean text
    clean_text = clean_data(string.lower())

    txt = ' '.join(clean_text)
    

    with open('classifier.pkl', 'rb') as classifier:
        model = pickle.load(classifier)
    
    vectorizer = CountVectorizer(stop_words='english')
    vector = vectorizer.transform([txt])
    #x = vectorizer.fit_transform(clean_text)
    prediction = model.predict(vector)

    print(prediction)

    #model.fit(x, )

    #y_pred = model.predict(x)

    #print(y_pred)

'''
negativeCount = 0
positiveCount = 0

for score in label_Predict:
    if score == '0':
        negativeCount += 1
    else:
        positiveCount += 1

label = 'Positive' if positiveCount>negativeCount else 'Negative'

print(label)
'''
text = "This is a great movie! Well done!"
predict_sentiment(text)
'''
parser = argparse.ArgumentParser(description="Predict sentiment of text")
parser.add_argument('text', metavar='text', type=str, help="Insert text to classify.")
args = parser.parse_args()
'''
