import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score
import numpy as np
stopwords = set(stopwords.words('english'))

# Define function to preprocess text
def preprocess_text(text):
    # If the input is a dictionary, get the text content
    if isinstance(text, dict):
        text = text.get('content', '')
    
    # Convert byte-like object to string
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Return preprocessed text
    return ' '.join(words)

# Define function to count words in a document
def count_words(text):
    # Split into words
    words = text.split()
    # Count occurrences of each word
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    # Return word count dictionary
    return counts

# Define function to represent documents as word frequency vectors
def create_word_frequency_vectors(docs):
    vectors=[]
    for doc in docs:
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform(doc)
        vectors.apppend(tfidf_vectors)
    return vectors




def cluster_news(ncluster,kmeans,whc,ac,dbscan):

    tar = tarfile.open("20_newsgroups.tar.gz", "r:gz")

    documents = []
    x=0
    for member in tar.getmembers():
        if member.isfile():
            f = tar.extractfile(member)
            content = f.read()
            try:
                # Try to decode the content as UTF-8
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # If decoding as UTF-8 fails, try decoding as UTF-16LE
                content = content.decode('utf-16le')
            # Convert the decoded string to bytes using UTF-8 encoding
            content = content.encode('utf-8')
            documents.append(content)
            x+=1
            
    print(x)
    tar.close()

    vectors = create_word_frequency_vectors(documents)
    arr=np.array(vectors)
    k=ncluster
    if (kmeans==True):
        kmeans = KMeans(k, random_state=0).fit(arr)
        predicted_k=kmeans.labels_
    if (whc==True):
        ward=AgglomerativeClustering(k,linkage='ward').fit(arr) 
        predicted_w=ward.labels_
    if (ac==True):
        aggl=AgglomerativeClustering(k).fit(arr)
        predicted_a=aggl.labels_
    if (dbscan==True):
        dbscan = DBSCAN(eps=0.5, min_samples=2).fit(arr)
        predicted_db=dbscan.labels_

    #calculate ground_truth labels for each,calculate and print them 
    #ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    #ars = adjusted_rand_score(true_labels, predicted_labels)
    #completeness = completeness_score(true_labels, predicted_labels)


parser = argparse.ArgumentParser()
parser.add_argument('--ncluster',type=int,default=20)
parser.add_argument('--kmeans', type=bool, default=True)
parser.add_argument('--whc', type=bool, default=False)
parser.add_argument('--ac', type=bool, default=False)
parser.add_argument('--dbscan',type=bool,default=False)
args = parser.parse_args()
