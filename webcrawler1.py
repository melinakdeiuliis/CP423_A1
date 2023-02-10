import hashlib
import requests
import re
import time
import argparse
'''
Text Retrieval and Search Engine - Assignment One
WebCrawler1.py 
'''

def crawler1(url, maxdepth, rewrite, verbose):
    h = hashlib.sha1(url.encode()).hexdigest()
    filename = h + '.txt'

    if not rewrite and file_exists(filename):
        return

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return

        with open(filename, 'w') as f:
            f.write(response.text)

        links = re.findall(r'<a\s.*?href=[\'"](.*?)[\'"]', response.text)

        with open('crawler1.log', 'a') as f:
            f.write('{},{},{},{}\n'.format(h, url, time.ctime(), response.status_code))

        if verbose:
            print('{},{}'.format(url, maxdepth))

        if maxdepth > 0:
            for link in links:
                crawler1(link, maxdepth - 1, rewrite, verbose)

    except:
        pass

def file_exists(filename):
    try:
        with open(filename):
            return True
    except:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initialURL', help='Initial URL to start crawling from')
    parser.add_argument('--maxdepth', type=int, help='Maximum number of depths to crawl from initial URL', required=True)
    parser.add_argument('--rewrite', type=bool, help='Re-download and re-write URL if file exists', default=False)
    parser.add_argument('--verbose', type=bool, help='Print URL and depth on the screen', default=False)
    args = parser.parse_args()

    crawler1(args.initialURL, args.maxdepth, args.rewrite, args.verbose)


#python webcrawler1.py --maxdepth=3 --rewrite=False --verbose=False https://crawler-test.com/links/nofollowed_page
