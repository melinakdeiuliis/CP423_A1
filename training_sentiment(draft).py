# Import libraries
import argparse
import json
import numpy as np
import string
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('English'))

# Import dataset
fileName = 'imdb_labelled.txt'

with open(fileName, 'r') as file:
    # Class counts
    yesCount = 0
    noCount = 0

    for line in file:
        # Tokenize line
        tokens = word_tokenize(line)

        print(tokens)

        clean_tokens = []
        # Remove stopwords and puncuation
        for token in tokens:
            if token.lower() not in stop_words and token not in string.punctuation:
                clean_tokens.append(token.lower)

            
        if clean_tokens[-1] == '1':
            yesCount += 1
        else:
            noCount += 1

        print(clean_tokens)
