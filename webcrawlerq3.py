import hashlib
import requests
import sys
import re


initialURL = "https://www.vedantu.com/english/cat-essay"

# Get html contents of the URL
response = requests.get(initialURL)
html_content = response.content

# Create file name of the hash value of URL using the hashlib library 
hash_object = hashlib.md5(initialURL.encode())
file_name = hash_object.hexdigest() + '.html'

# Use the requests library to download the HTML contents of the URL
response = requests.get(initialURL)
html_content = response.content
html_text = response.text

# Write the HTML contents to a file with the hashed name
with open(file_name, 'wb') as f:
    f.write(html_content)

# soup = BeautifulSoup(html_content)
# print(soup.get_text)

html_tag = re.compile(b'<.*?>')

with open(file_name, 'rb') as html_file:
    line = html_file.readline()
    while line:
        if html_tag.search(line):
            line = re.sub(b'<.*?>', '1', line)

print(html_text)
            









