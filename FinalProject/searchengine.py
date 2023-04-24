"""
Search Engine Project 
CP423 - April 2023
Anastasia, Melina and Ryan
"""
import soundex

class InvertedIndex:
    def __init__(self):
        self.index = {}
        
    def add_document(self, doc_hash, text):
        for word in text.split():
            soundex_code = soundex.soundex(word)
            if soundex_code not in self.index:
                self.index[soundex_code] = {}
            if doc_hash not in self.index[soundex_code]:
                self.index[soundex_code][doc_hash] = 0
            self.index[soundex_code][doc_hash] += 1
            
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
        if (opt==1):
            print("reading source.txt and populating data folder")
            with open("sources.txt", 'r') as f:
                for line in f:
                    topic,url=line.strip().split(",")
                    topic_folder = os.path.join("Data", topic)
                    response=requests.get(url)
                    content=response.text
                    h = hashlib.sha1(url.encode()).hexdigest()
                    filename = h + '.txt'
                    fn=os.path.join(topic_folder,filename)
                    with open(fn,'w',encoding='utf-8') as output:
                        output.write(content)

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
