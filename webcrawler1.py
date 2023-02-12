'''
Text Retrieval and Search Engine - Assignment One
WebCrawler1.py 
Anastasia, Melina & Ryan 
'''
import requests
import os
import hashlib
import re
import time
import argparse

def webcrawler1(url, maxdepth, rewrite, verbose):
    h = hashlib.sha1(url.encode()).hexdigest()
    filename = h + '.txt'
    if  os.path.isfile(filename) and not rewrite:
        return
    try:
        response = requests.get(url)

        with open(filename, 'w') as f:
            f.write(response.text)

        link = re.findall(r'<a.*?href=[\'"](.*?)[\'"]', response.text)

        with open('crawler1.log', 'a') as f:
            f.write('{},{},{},{}\n'.format(h, url, time.ctime(), response.status_code))

        if (verbose==True):
            print('{}, {}'.format(url, maxdepth))

        if (maxdepth > 0):
            for l in link:
                webcrawler1(l, maxdepth - 1, rewrite, verbose)
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initialURL', help='Initial URL to start crawling from')
    parser.add_argument('--maxdepth', type=int, required=True)
    parser.add_argument('--rewrite', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    webcrawler1(args.initialURL, args.maxdepth, args.rewrite, args.verbose)


#python webcrawler1.py --maxdepth=5 --rewrite=False --verbose=False https://crawler-test.com/links/relative_link/a/b
#python webcrawler1.py --maxdepth=5 https://crawler-test.com/links/nofollowed_page
