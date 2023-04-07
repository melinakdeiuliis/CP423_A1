import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import os
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
    words = [word for word in words if word not in stopwords]
    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Return preprocessed text
    return ' '.join(words)


# Define function to represent documents as word frequency vectors
def create_word_frequency_vectors(docs):
    vectors=[]
    for doc in docs:
        doc=preprocess_text(doc)
        vectorizer = TfidfVectorizer()
        vectorizer.fit([doc])
        tfidf_vector = vectorizer.transform([doc])
        vectors.append(tfidf_vector)
        #print(tfidf_vector.toarray())
    return vectors

def cluster_news(ncluster,kmeans,whc,ac,dbscan):

    # open and read dataset
    # filename = "20_newsgroups.tar.gz"
    # tar = tarfile.open("20_newsgroups.tar.gz", "r:gz")
    # if not os.path.exists("20_newsgroups"):
    #     with tarfile.open(filename, "r:gz") as tar:
    #         tar.extractall()
    # #create array of strings where each string is a document
    # documents = []
    # newsgroup_names = os.listdir("20_newsgroups")
    # for newsgroup_name in newsgroup_names:
    #     newsgroup_path = os.path.join("20_newsgroups", newsgroup_name)
    #     if os.path.isdir(newsgroup_path):
    #         for filename in os.listdir(newsgroup_path):
    #             file_path = os.path.join(newsgroup_path, filename)
    #             with open(file_path, "r", encoding="ISO-8859-1") as f:
    #                 document = f.read()
    #                 documents.append(document)
    # tar.close()
    #create vectors


    documents=["The city was the administrative centre of Chernobyl Raion (district) from 1923. After the disaster, in 1988, the raion was dissolved and administration was transferred to the neighbouring Ivankiv Raion. The raion was abolished on 18 July 2020 as part of the administrative reform of Ukraine, which reduced the number of raions of Kyiv Oblast to seven. The area of Ivankiv Raion was merged into Vyshhorod Raion."]
    vectors = create_word_frequency_vectors(documents)
    
    arr=np.array(vectors)
    for k in ncluster:
        if (kmeans==True):
            kmeans = KMeans(k, random_state=0).fit(arr)
            predicted_k=kmeans.labels_
            print("Adjusted Mutual Information Score:") #{}".format(adjusted_mutual_info_score(true_labels, predicted_k)))
            print("Adjusted Random Score: ")#{}".format(adjusted_rand_score(true_labels, predicted_k))
            print("Completeness Score: ")#{}".format(completeness_score(true_labels, predicted_k))
        if (whc==True):
            ward=AgglomerativeClustering(k,linkage='ward').fit(arr) 
            predicted_w=ward.labels_
            print("Adjusted Mutual Information Score:") #{}".format(adjusted_mutual_info_score(true_labels, predicted_w)))
            print("Adjusted Random Score: ")#{}".format(adjusted_rand_score(true_labels, predicted_w))
            print("Completeness Score: ")#{}".format(completeness_score(true_labels, predicted_w))
        if (ac==True):
            aggl=AgglomerativeClustering(k).fit(arr)
            predicted_a=aggl.labels_
            print("Adjusted Mutual Information Score:") #{}".format(adjusted_mutual_info_score(true_labels, predicted_a)))
            print("Adjusted Random Score: ")#{}".format(adjusted_rand_score(true_labels, predicted_a))
            print("Completeness Score: ")#{}".format(completeness_score(true_labels, predicted_a))
        if (dbscan==True):
            dbscan = DBSCAN(eps=0.5, min_samples=2).fit(arr)
            predicted_db=dbscan.labels_
            print("Adjusted Mutual Information Score:") #{}".format(adjusted_mutual_info_score(true_labels, predicted_db)))
            print("Adjusted Random Score: ")#{}".format(adjusted_rand_score(true_labels, predicted_db))
            print("Completeness Score: ")#{}".format(completeness_score(true_labels, predicted_db))


        print("Predicted kmeans labels: {}".format(predicted_k))
        print("Predicted ward labels: {}".format(predicted_w))
        print("Predicted agglomerative labels: {}".format(predicted_a))
        print("Predicted dbscan labels: {}".format(predicted_db))


    #calculate ground_truth labels for each,calculate and print them 
    #ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    #ars = adjusted_rand_score(true_labels, predicted_labels)
    #completeness = completeness_score(true_labels, predicted_labels)




parser = argparse.ArgumentParser()
parser.add_argument('--ncluster',type=int,nargs = '+',default=[20])
parser.add_argument('--kmeans', type=bool, default=True)
parser.add_argument('--whc', type=bool, default=False)
parser.add_argument('--ac', type=bool, default=False)
parser.add_argument('--dbscan',type=bool,default=False)
args = parser.parse_args()

cluster_news(args.ncluster,args.kmeans,args.whc,args.ac,args.dbscan)
