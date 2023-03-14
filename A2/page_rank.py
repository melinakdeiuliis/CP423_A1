import argparse
from bs4 import BeautifulSoup
import gzip 
def summation(node,pageranks,nodes):
    x=10 
    return
def page_rank(max,lambda,thr,nodes):
    with gzip.open('web-Stanford.txt.gz', 'rb') as f:
        file_content = f.read()
    file_content = f.read().decode('utf-8')
    pagerank=(lambda/len(nodes))+(1-lambda)

parser = argparse.ArgumentParser()
parser.add_argument("--maxiteration",type=int)
parser.add_argument("--lambda",type=float)
parser.add_argument("--thr",type=float)
parser.add_argument("--nodes")
