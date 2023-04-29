import hashlib
import datetime
import os
import requests
from bs4 import BeautifulSoup
import justext
from urllib.parse import urlparse

# Read sources.txt file and store the URLs and topics in a list
with open('sources.txt', 'r') as f:
    lines = f.read().splitlines()

urls = []
topics = []

for line in lines:
    topic, url = line.split(',', 1)
    topics.append(topic.strip())
    urls.append(url.strip())

# Create a dictionary to store the topic related subfolders
subfolders = {}

# Define a function to crawl a URL and extract its content and links
def crawl(url, topic):
    try:
        # Create a subfolder for the topic if it doesn't exist
        if topic not in subfolders:
            subfolders[topic] = True
            os.makedirs(f"data/{topic}", exist_ok=True)
        # Check if the page is available in the related subfolder
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        file_name = os.path.join(f"data/{topic}", url_hash+'.txt')
        if os.path.isfile(file_name):
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            # Parse the cached file to extract links
            soup = BeautifulSoup(content, 'html.parser')
        else:
            # Crawl the page
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract the content using justext
            paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
            content = '\n\n'.join([p.text for p in paragraphs])
            # Save the content in a file
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
        # Write the <topic, linkâ€™s URL, Hash value of URL, date> in crawl.log file
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('crawl.log', 'a', encoding='utf-8') as f:
            f.write(f"{topic}, {url}, {url_hash}, {date}\n")
        # Crawl the links in the page if they belong to the same domain as the initial URL
        domain = urlparse(url).netloc
        
        if soup:
            for link in soup.find_all('a'):
                link_url = link.get('href')
                if link_url and urlparse(link_url).netloc == domain:
                    crawl(link_url, topic)
    except Exception as e:
        print(f"Error while crawling {url}: {e}")

# Crawl the URLs in sources.txt
for url, topic in zip(urls, topics):
    # Crawl the URL
    crawl(url, topic)