import argparse
import json
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#noisy channel matrix
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    matrix = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        matrix[i][0] = i
    for j in range(1, n+1):
        matrix[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = 1 + min(matrix[i][j-1], matrix[i-1][j], matrix[i-1][j-1])
    return matrix[m][n]
def suggest_correction(word, tokens):
    best_word = None
    best_distance = float('inf')
    for token in tokens:
        distance = edit_distance(word, token)
        if distance < best_distance:
            best_word = token
            best_distance = distance
    return best_word
def noisy_channel(correct,proba,values):
    with open("data_wikipedia/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json","r",encoding='utf-8') as f:
        data=json.load(f)
    f.close()
 #   Create tokens word for word
    invertedindex={}
    total=0
    for entry in data:
        for word in (nltk.word_tokenize(entry["text"])):
            if word not in stop_words:
                if (word in invertedindex):
                    invertedindex[word]+=1
                else:
                    invertedindex[word]=1
                total+=1
    word_list = values.split(',')
    if correct:
        #call correct func
        for x in word_list:
            print('Word: {}  Correction: {}'.format(x,suggest_correction(x,invertedindex)))
    if proba:
        #call probability func
        for x in word_list: 
            if (x in invertedindex):
                print('Word: {}  Probability(Word): {}'.format(x,invertedindex[x]/total))
            else:
                new_word=suggest_correction(x,invertedindex)
                print("Word: {} CorrectedWord: {} Probability(Word): {}".format(x,new_word,invertedindex[new_word]/total))
            
parser = argparse.ArgumentParser()
parser.add_argument('--correct',type=bool)
parser.add_argument("--proba",type=bool)
parser.add_argument("--values")
args = parser.parse_args()
noisy_channel(args.correct,args.proba,args.values)

