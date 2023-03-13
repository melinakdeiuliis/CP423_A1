import argparse
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
stop_words = set(stopwords.words('english'))

def noisy_channel(correct,proba):
    with open("data_wikipedia/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json","r",encoding='utf-8') as f:
        data=json.load(f)

    f.close()
    #Create tokens word for word
    tokenize=[]
    for entry in data:
        for word in (nltk.word_tokenize(entry["text"])):
            if word.lower() not in stop_words:
                tokenize.append(word)











parser = argparse.ArgumentParser()
parser.add_argument('--correct',type=bool,default=False)
parser.add_argument('--proba', type=bool, default=False)

#parser.add_argument("values",type=)
#type array?^

args = parser.parse_args()
noisy_channel(args.correct,args.proba)