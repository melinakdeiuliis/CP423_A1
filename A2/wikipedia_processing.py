import argparse
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
stop_words = set(stopwords.words('english'))
'''
processes wikipedia articles in the form of .json 
Parameters:
    zipf - boolean, if true will display a diagram for rank vs. frequency
    tokens - boolean, if true will write all tokens to wikipedia.token
    stop - boolean, if true will write all tokens withou stop words to wikipedia.token.stop
    stem - boolean, if true will write all stemmed tokens to wikipedia.token.stemm
    inverted- boolean, if true will create an inverted index counting number of occurance of tokens
Returns: 
    Generated files from options under parameters if any are true
'''
def wikipedia_processing(zipf,tokens,stop,stem,inverted):
        
    with open("data_wikipedia/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json","r",encoding='utf-8') as f:
        data=json.load(f)

    f.close()
    # Tokenize Once for optimization
    tokenizer=[]
    no_stop_words=[]
    stemmed_tokens=[]
    for entry in data:
        for word in (nltk.word_tokenize(entry["text"])):
            tokenizer.append(word)
            stemmed_tokens.append(PorterStemmer().stem(word))
            if word.lower() not in stop_words:
                no_stop_words.append(word)
    if (zipf or inverted):
        #calculate counts/frequencies for tokens once
        inverted_index={}
        for token in tokenizer:
            if (token in inverted_index):
                inverted_index[token]+=1
            else:
                inverted_index[token]=1
        if (zipf):
            sorted_tokens = sorted(inverted_index, key=inverted_index.get, reverse=True)
            frequencies = np.array([inverted_index[token] for token in sorted_tokens])
            ranks = np.arange(1, len(sorted_tokens)+1)
            plt.plot(ranks, frequencies, 'ro')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.title('Zipf Diagram')
            plt.show()
    if (tokens):
        with open('wikipedia.token', 'w',encoding='utf-8') as f:
            for token in tokenizer:
                f.write(token + '\n')
    if (stop):
        with open('wikipedia.token.stop', 'w',encoding='utf-8') as f:
            for token in no_stop_words:
                f.write(token+'\n')
    if (stem):
        with open('wikipedia.token.stemm', 'w',encoding='utf-8') as f:
            for token in stemmed_tokens:
                f.write(token+'\n')
                
parser = argparse.ArgumentParser()
parser.add_argument('--zipf',type=bool,default=False),
parser.add_argument('--tokenize', type=bool, default=False)
parser.add_argument('--stopword', type=bool, default=False)
parser.add_argument('--stemming', type=bool, default=False)
parser.add_argument('--invertedindex',type=bool,default=False)
args = parser.parse_args()

wikipedia_processing(args.zipf,args.tokenize,args.stopword,args.stemming,args.invertedindex)
