from scholarly import scholarly
import requests

title = '"A key area of knowledge delivered by someone knowledgeable": Feminist expectations and explorations of a one-off economics lecture on gender'
search_query = scholarly.search_pubs_query(title)
pub = next(search_query)
pdf_url = pub['eprint']
print(pub)

r = requests.get(pdf_url)
with open('./test.pdf', 'wb') as f:
    f.write(r.content)



