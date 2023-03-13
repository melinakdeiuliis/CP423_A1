from urllib.request import urlopen

from bs4 import BeautifulSoup

import sys 

import requests

# --maxiteration: the maximum number of iterations to stop if algorithm has not converged 
# -- lamda: the λ paramter value 
# -- thr: the threshold value
# -- nodes: the NodeIDs that we want to get their PageRank values at the end of iterations 

html =urlopen('https://snap.stanford.edu/data/web-Stanford.html')

max_iteration = sys.argv[1]

lamda = sys.argv[2]

thr = sys.argv[3]

node = sys.argv[4]
 

