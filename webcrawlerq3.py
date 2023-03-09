import hashlib
import requests
import sys
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def func(i, j):
    return i * j
def webcrawler3(initialURL):
    
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

    soup=BeautifulSoup(html_text,"html.parser")
    # Replace html tags with 1
    for tag in soup.find_all():
        tag.replace_with("1")
    # Replace tokens with 0 using regex
    tokens=re.compile(r'\S+')
    new_html=tokens.sub('0',str(soup))

    '''Smaller html file to reduce amount of data for processing graph'''
    #n=len(new_html)

    '''Full html file '''
    n=len(html_text)

    # Create matrix for plotting
    matrix=[[0 for j in range(n)]for i in range(n)]
    for i in range(n):
        for j in range(i,n):
            b=html_text[i:j+1].count('1')
            matrix[i][j]=b+(j-i+1-2*b)

    ''' Display plot as 2D heat map'''
    plt.imshow(matrix,cmap="autumn",interpolation='nearest')
    plt.show()

    '''Display plot as 3D heat map'''
    # n=len(matrix)
    # x=np.arange(n)
    # y = np.arange(n)
    # x = np.arange(n).reshape((-1,1))
    # y = np.arange(n).reshape((-1,1))
    # X,Y=np.meshgrid(x,y)
    # Z=np.zeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         if (matrix[i][j]==1):
    #             Z[i][j]=1
    #         else:
    #             Z[i][j]=0
    #         Z[i][j]=func(i,j)
    # fig=plt.figure()
    # axis = fig.add_subplot(111, projection='3d')
    # axis.plot_surface(X,Y,Z, cmap='viridis')
    # axis.set_xlabel('i')
    # axis.set_ylabel('j')
    # axis.set_zlabel('f(i,j)')
    # plt.show()


    # Save main content to txt
    content=html_text[0:n-1]
    new_filename=hash_object.hexdigest()+".txt"
    with open(new_filename,"w") as f:
        f.write(content)

if __name__ == '__main__':
    url=sys.argv[1]
    webcrawler3(url)
