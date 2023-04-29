import os
import re
import hashlib
import jellyfish

# Define the list of topics and their corresponding folder names
topics = {
    'Technology': 'data/Technology',
    'Lifestyle': 'data/Lifestyle',
    'Astronomy': 'data/Astronomy'
}

# Define the name of the output files
index_file = 'invertedindex.txt'
mapping_file = 'mapping.txt'

# Define a function to generate the soundex code for a term
def get_soundex(term):
    soundex_code = jellyfish.soundex(term)
    return soundex_code

# Define a function to generate the hash value for a document
def get_hash(doc):
    return hashlib.md5(doc.encode('utf-8')).hexdigest()

#  index dictionary
index = {}

# mapping dictionary
mapping = {}

# doc counter
doc_counter = 0

# Loop through each topic
for topic, folder in topics.items():
    # Loop through each document in the folder
    for filename in os.listdir(folder):
        # Read the contents of the document
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate the hash value for the document
        doc_hash = get_hash(content)

        if doc_hash not in mapping:
            # Add the mapping between the doc hash and doc ID
            mapping[doc_hash] = doc_counter
            doc_id = doc_counter
            
            # Split the content into terms
            terms = re.findall(r'\b\w+\b', content.lower())
            
            # Loop through each term and add it to the index
            for term in terms:
                # Generate the soundex code for the term
                soundex_code = get_soundex(term)
                
                # Add the term to the index
                if term not in index:
                    index[term] = {}
                if soundex_code not in index[term]:
                    index[term][soundex_code] = []
                if (doc_id, 1) not in index[term][soundex_code]:
                    index[term][soundex_code].append((doc_id, 1))
            
            # Increment the doc counter
            doc_counter += 1

# Write the index to the output file
with open(index_file, 'w', encoding='utf-8') as f:
    f.write('| Term | Soundex | Appearances (DocID, Frequency) |\n')
    f.write('|------|---------|--------------------------------|\n')

    for term in sorted(index.keys()):
        for soundex_code in sorted(index[term].keys()):
            doc_freq = len(index[term][soundex_code])
            postings = ' '.join([f'({doc_id}, {freq})' for (doc_id, freq) in index[term][soundex_code]])
            f.write(f'| {term} | {soundex_code} | {doc_freq} {postings} |\n')

# Write the mapping to the output file
with open(mapping_file, 'w', encoding='utf-8') as f:
    for doc_hash in sorted(mapping.keys()):
        f.write(f'{doc_hash} {mapping[doc_hash]}\n')