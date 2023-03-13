# CP423_A1
# Text Retreival and Search Engine 

AssignmentOne: There are three different types of web crawlers stored in assignment one which are described under the course assignment details. 1-downloads and saves content while searching through different hyperlinks to introduce depth to search engines. 2-Extracts information about a scholarly researcher on Google scholar page. 3-Seperates all html tags and tokens and is then represented in a 2d or 3d heat map plotting f(i,j).
AssignmentTwo: There are 4 modules in this assignment that encapture natural language processing inside search engines by using different algorithms such as zipf law, stemming, page rank, encoding/decoding and more. 1 - downloads a .json file from wikipedia dataset and tokenizes the file for further information processing. 2 - encodes and decodes using elias gamma and delta. 3 - Calculates page rank from a stanford web graph. 4 - creates a spell checking script to recommend correctly spelt words.

## Authors

- [@ana-mali](https://www.github.com/ana-mali)
- [@melinakdeiuliis](https://github.com/melinakdeiuliis)
- [@ryanayala](https://github.com/ryanayala)

## Installation

For Assignment One;
Make sure you have all python libraries installed to properly run the webcrawlers which are stated at the top of each module. The following commands can be used in your cmd for some important libraries used.
```bash
  $ pip install requests 

```
```bash
  $ pip3 install matplotlib

```
```bash
  $ pip install beautifulsoup4

```
For Assignment Two; 
The most important library to have installed is nltk. Make sure you have nltk installed on your computer before running the programs as you may need to add additional code in the python module to then install more specific packages. 
```bash
  $ pip install nltk

```
If you are having problems running the code, please edit the python module and add a new line of code under the imports the following lines for question one;
```bash
  $ nltk.download('punkt')

```
```bash
  $ nltk.download('stopwords')

```
