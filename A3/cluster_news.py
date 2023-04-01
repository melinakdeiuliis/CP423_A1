import argparse
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import urllib.request
import tarfile
stopwords = set(stopwords.words('english'))

# Define function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
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
    # Preprocess documents and count words
    word_counts = [count_words(preprocess_text(doc)) for doc in docs]
    # Create dictionary of unique words
    unique_words = sorted(list(set(word for word_count in word_counts for word in word_count.keys())))

    # Create word frequency vectors
    vectors = []
    for word_count in word_counts:
        vector = [word_count.get(word, 0) for word in unique_words]
        vectors.append(vector)

    # Return word frequency vectors and unique words
    return vectors, unique_words


tar = tarfile.open("20_newsgroups.tar.gz", "r:gz")

documents = []
x=0
for member in tar.getmembers():
    if member.isfile():
        f = tar.extractfile(member)
        content = f.read()#.decode("utf-16le")
        documents.append(content)
        x+=1
        
print(x)
tar.close()

vectors, unique_words = create_word_frequency_vectors(documents)
print('Unique words:', unique_words)
print('Word frequency vectors:')
for vector in vectors:
    print(vector)


parser = argparse.ArgumentParser()
parser.add_argument('--ncluster',type=bool,default=False),
parser.add_argument('--kmeans', type=bool, default=False)
parser.add_argument('--whc', type=bool, default=False)
parser.add_argument('--ac', type=bool, default=False)
parser.add_argument('--dbscan',type=bool,default=False)
args = parser.parse_args()
