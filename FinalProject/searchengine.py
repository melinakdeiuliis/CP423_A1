"""
Search Engine Project 
CP423 - April 2023
Anastasia, Melina and Ryan
"""

def printoptions():
    print("""Select an option:
1- Collect new documents.
2- Index documents.
3- Search for a query.
4- Train ML classifier.
5- Predict a link.
6- Your story!
7- Exit""")
    
    

def searchengine():
    printoptions()
    opt=input("Please select an option by inputting an integer: ")
    opt=int(opt)
    
    while (opt!=7):
        """Collect new documents
    read sources.txt and create hash.txt for each site"""
        if (opt==1):
            print("reading source.txt")





        """
        Index documents
        create invertedindex.txt
        """
        elif(opt==2):
            print("creating invertedindex.txt")
        



        
        """
        Search for query
        input prompt to print top 3 related documents
        Might want to use inverted index
        """
        elif(opt==3):
            query=input("Please enter query: ")
        



        '''
        Train ML Classifier
        Pick best classifier, train and save it as classifier.model and print results
        '''
        elif(opt==4):




        '''
        Predict a link
        crawl,extract,vectorize and print predicted label + confidence of classifier
        '''
        elif(opt==5):
            link=input("Please enter a link: ")
        


        '''
        your story
        create a story.txt and print it 
        '''
        elif(opt==6):


        else:
            print("Wrong input please try again. Must be integer 1-7")
        
        printoptions()
        opt=input("Please select an option by inputting an integer: ")
        opt=int(opt)
searchengine()
