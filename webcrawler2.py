import hashlib
from bs4 import BeautifulSoup
import requests
import sys
import json



def webcrawler2(url):
    #Extract url information
    response = requests.get(url)
    html_content = response.content
    hash_object = hashlib.md5(url.encode())
    filename = hash_object.hexdigest() + '.txt'
    #Download content from url onto .txt file
    with open(filename, 'w') as f:
            f.write(response.text)
    #Extract researcher information
    print("I AM HERE")
    soup = BeautifulSoup(html_content, "html.parser")
    researcher_data=[]
    researcher_name = soup.find("div", {"class": "gsc_prf_in"}).get_text()
    researcher_caption = soup.find("div", {"class": "gsc_prf_il"}).text
    researcher_institution = soup.find("div", {"class": "gsc_prf_il"}).text
    researcher_data.append(researcher_name)
    researcher_data.append(researcher_caption)
    researcher_data.append(researcher_institution)
    researcher_keywords = [keyword.text for keyword in soup.find_all("a", {"class": "gsc_prf_inta"})]
    researcher_data.append(researcher_keywords)
    researcher_imgURL = soup.find("img", {"class": "gsc_prf_pimg"})["src"]
    researcher_data.append(researcher_imgURL)
    all_citations = soup.find("td", {"class": "gsc_rsb_std"}).text
    since2018_citations = soup.find("td", {"class": "gsc_rsb_std", "data-field": "citation_2018"}).text
    researcher_citations = {"all": all_citations, "since2018": since2018_citations}
    researcher_data.append(researcher_citations)
    all_hindex = soup.find("td", {"class": "gsc_rsb_std"}).text
    since2018_hindex = soup.find("td", {"class": "gsc_rsb_std", "data-field": "hindex_2018"}).text
    researcher_hindex = {"all": all_hindex, "since2018": since2018_hindex}
    researcher_data.append(researcher_hindex)
    all_i10index = soup.find("td", {"class": "gsc_rsb_std"}).text
    since2018_i10index = soup.find("td", {"class": "gsc_rsb_std", "data-field": "i10index_2018"}).text
    researcher_i10index = {"all": all_i10index, "since2018": since2018_i10index}
    researcher_data.append(researcher_i10index)
    coauthors = []
    for coauthor in soup.find_all("tr", {"class": "gsc_a_tr"}):
        coauthor_name = coauthor.find("a", {"class": "gsc_a_at"}).text
        coauthor_title = coauthor.find("div", {"class": "gsc_a_t"}).text
        coauthor_link = coauthor.find("a", {"class": "gsc_a_at"})["href"]
        coauthors.append(coauthor_name,coauthor_title,coauthor_link)
    researcher_data.append(coauthors)
    researcher_papers = []
    for paper in soup.find_all("tr", class_="gs_r"):
        paper_title = paper.find("a").get_text()
        paper_authors = paper.find("div", class_="gs_a").get_text()
        paper_journal = paper.find("div", class_="gs_rs").get_text()
        paper_citedby = paper.find("a", class_="gs_fl").get_text()
        researcher_papers.append(paper_title,paper_authors,paper_journal,paper_citedby)
    researcher_data.append(researcher_papers)
    #Put researcher content to .json file
    jsonfilename=hash_object+".json"
    with open(jsonfilename,'w') as file:
        json.dump(researcher_data,file)
    print("Hello")


if __name__ == '__main__':
    url=sys.argv[1]
    webcrawler2(url)
# python webcrawler2.py https://scholar.google.com/citations?user=4hwc6fwAAAAJ"&"hl=en"&"oi=ao
# #https://scholar.google.com/citations?user=iipZX04AAAAJ&hl=en&oi=ao