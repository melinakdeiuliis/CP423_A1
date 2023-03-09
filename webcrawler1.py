'''
Text Retrieval and Search Engine - Assignment One
WebCrawler1.py 
Anastasia, Melina & Ryan 
'''
import hashlib
import requests
import os
import time
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
depth=1
'''
WebcrawlerOne
Parameters:
    url - string with address of website to search 
    Maxdepth - integer extent to which it will index the sites content
    rewrite - boolean if true will re-download and re-write url
    verbose - boolean if true will print url and its depth on console
Returns:
    Creates hash.txt file that stores downloaded content from url
    Creates crawler1.log file that includes hash, url, datetime and http response code
    If verbose is true will return every link traversed along with its depth printed to screen
'''
def webcrawler1(url, maxdepth, rewrite, verbose,depth):
    if depth>maxdepth:
        return
    h = hashlib.sha1(url.encode()).hexdigest()
    filename = h + '.txt'
    response = requests.get(url)
    if  os.path.isfile(filename) and not rewrite:
        return
    elif response.status_code==200: 
        with open(filename, 'w',encoding="utf-8") as f:
            f.write(response.text)

        soup=BeautifulSoup(response.text,'html.parser')
        link=[]
        for links in soup.find_all('a'):
            href = links.get('href')
            if href:
                rel_url=urljoin(url,href)
                link.append(rel_url)
    
        with open('crawler1.log', 'a') as f:
            f.write('{},{},{},{}\n'.format(h, url, time.ctime(), response.status_code))
        
        if (verbose==True):
            print('{}, {}'.format(url, depth))

        #depth crawl
        for l in link:
            depth=depth+1
            webcrawler1(l, maxdepth, rewrite, verbose,depth+1)
            if (depth==maxdepth):
                break

#Create options 
parser = argparse.ArgumentParser()
parser.add_argument('initialURL')
parser.add_argument('--maxdepth', type=int, required=True)
parser.add_argument('--rewrite', type=bool, default=False)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--current_depth',type=int,default=1)
args = parser.parse_args()

webcrawler1(args.initialURL, args.maxdepth, args.rewrite, args.verbose,args.current_depth)
