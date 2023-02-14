import sys
import time
import requests
from bs4 import BeautifulSoup
import hashlib
import json


if len(sys.argv) != 2:
    print("Usage: python webcrawler3.py researcherURL")
    sys.exit()


researcherURL = sys.argv[1]


response = requests.get(researcherURL)


# Calculate the hash of the researcher Google Scholar page
hash_object = hashlib.md5(researcherURL.encode())
hash_value = hash_object.hexdigest()


#H.txt
with open(f"{hash_value}.txt", "w") as f:
    f.write(response.text)


# Parse the content of the researcher's Google Scholar page using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')


# Extract the information from the page
name = soup.find(id="gsc_prf_in")
if name:
    name = name.contents
else:
    name = "N/A"

caption = soup.find("a", class_="gsc_a_at")
if caption:
    caption = caption.text
else:
    caption = "N/A"

institution = soup.find("a", class_="gsc_prf_ila")
if institution:
    institution = institution.text
else:
    institution = "N/A"

department = soup.find("div", class_="gsc_prf_il")
if department:
    department = department.text
else:
    department = "N/A"

keywords = [tag.text for tag in soup.find_all("a", class_="gsc_prf_inta gs_ibl")]

img_url = soup.find("img", id="gsc_prf_pup-img")
if img_url:
    img_url = img_url["src"]
else:
    img_url = "N/A"

citations_all = soup.find_all("td", class_="gsc_rsb_std")
if len(citations_all) > 1:
    citations_since2018 = citations_all[1].text
else:
    citations_since2018 = "N/A"

hindex_all = soup.find_all("td", class_="gsc_rsb_std")
if len(hindex_all) > 4:
    hindex = hindex_all[4].text
else:
    hindex = "N/A"

hindex_since2018 = hindex_all[3].text if len(hindex_all) > 3 else "N/A"


i10index_all = soup.find_all("td", class_="gsc_rsb_std")
i10index_since2018 = i10index_all[5].text if len(i10index_all) > 5 else "N/A"


coauthors = []
for row in soup.find_all("div", class_="gsc_rsb_title"):
    coauthor_name = row.find("h3", class_="gsc_rsb_a_desc").text
    coauthor_title = row.find("div", class_="gs_ggsd").text if row.find("div", class_="gs_ggsd") else "N/A"
    coauthor_link = row.find("a")["href"]
    coauthors.append({
        "coauthor_name": coauthor_name,
        "coauthor_title": coauthor_title,
        "coauthor_link": coauthor_link
    })



# Extract paper information
papers = []
for row in soup.find_all("div", class_="gs_wr"):
    title = row.select_one('.gsc_a_a').text
    authors = row.select_one('.gs_gray').text
    venue = row.select_one('.gs_gray').find_next_sibling('div').text
    cited_by = row.select_one('.gs_nph').text
    year = row.select_one('.gsc_a_a').text
    papers.append({
        'paper_title': title,
        'paper_authors': authors,
        'paper_journal': venue,
        'paper_citedby': cited_by,
        'paper_year': year
    })


# Check if the "Show More" button exists on the page
show_more_button = soup.find("button", id="gsc_bpf_more")
if show_more_button:
    
    while True:
       
        cid = show_more_button["data-cid"]
        
        post_data = {"cstart": len(papers), "pagesize": 100, "citation_for_view": cid, "hl": "en", "xhr": "t"}
        ajax_headers = {"Referer": researcherURL}
        ajax_response = requests.post("https://scholar.google.com/citations", data=post_data, headers=ajax_headers)
        # Parse the AJAX response 
        ajax_soup = BeautifulSoup(ajax_response.content, 'html.parser')
        # Extract the paper information from the AJAX response
        ajax_papers = []
        for row in ajax_soup.select("tr.gsc_a_tr"):
            title = row.select_one('.gsc_a_at').text
            authors = row.select_one('.gs_gray').text
            venue = row.select_one('.gs_gray').find_next_sibling('div').text
            cited_by = row.select_one('.gsc_a_ac').text
            year = row.select_one('.gs_scl').text
            ajax_papers.append({
                'paper_title': title,
                'paper_authors': authors,
                'paper_journal': venue,
                'paper_citedby': cited_by,
                'paper_year': year
            })
        papers += ajax_papers
        # Check if more papers to load
        show_more_button = ajax_soup.find("button", id="gsc_bpf_more")
        if not show_more_button:
            break



output = {
    'researcher_name': name,
    'researcher_caption': caption,
    'researcher_institution': institution,
    'researcher_department': department,
    'researcher_keywords': keywords,
    'researcher_imgURL': img_url,
    'researcher_citations': {
        'all': citations_all,
        'since2018': citations_since2018
    },
    'researcher_hindex': {
        'all': hindex_all,
        'since2018': hindex_since2018
    },
    'researcher_i10index': {
        'all': i10index_all,
        'since2018': i10index_since2018
    },
    'researcher_coauthors': coauthors,
    'researcher_papers': papers
}


# Write output to file
with open(f'{hash_value}.json', 'w') as f:
    json.dump(output, f, indent=4)

