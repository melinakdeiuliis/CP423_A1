import hashlib
import requests
import sys
import re
import matplotlib.pyplot as plt

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

    # Replace html tags with 1
    html_text=re.sub(r'<.*?>','1',html_text)
    # Replace tokens with 0
    html_text=re.sub(r'\n\w+\b','0',html_text)

    n=len(html_text)
    # Create matrix for plotting
    matrix=[[0 for j in range(n)]for i in range(n)]
    for i in range(n):
        for j in range(i,n):
            b=html_text[i:j+1].count('1')
            matrix[i][j]=b+(j-i+1-2*b)

    # Display plot as 2D heat map
    #plt.imshow(matrix,cmap="autumn",interpolation='nearest')
    #plt.show()

    # Display plot as 3D heat map
    G=plt.figure()
    x = [i for i in range(len(matrix)) for j in range(len(matrix[0]))]
    y = [j for i in range(len(matrix)) for j in range(len(matrix[0]))]
    z = [matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0]))]
    axis=G.add_subplot(111,projection='3d')
    graph=axis.scatter(x,y,z)
    axis.set_xlabel('i')
    axis.set_ylabel('j')
    axis.set_zlabel('f(i,j)')
    plt.show()


    # Save main content to txt
    content=html_text[i:j+1]
    new_filename=hash_object.hexdigest()+".txt"
    with open(new_filename,"w") as f:
        f.write(content)

if __name__ == '__main__':
    #initialURL = "https://www.vedantu.com/english/cat-essay"
    url=sys.argv[1]
    webcrawler3(url)
    #python webcrawler3.py https://crawler-test.com/content/word_count_100_words
