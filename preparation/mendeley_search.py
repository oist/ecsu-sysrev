import requests
import json
from commons import MODS_SENSES_COMBO_STR, validate_paper
from tqdm import tqdm

from key import MENDELEY_ACCESS_TOKEN
# expires every hour, get new tokens: https://mendeley-show-me-access-tokens.herokuapp.com/

MENDELEY_API_BASE = 'https://api.mendeley.com/search/catalog'
# Query terms, which match any field in the document, e.g. title, author etc (required).
# https://dev.mendeley.com/methods/#http-request35

MENDELEY_HEADERS = {
	'Authorization': 'Bearer {}'.format(MENDELEY_ACCESS_TOKEN),
	'Accept': 'application/vnd.mendeley-document.1+json'
}
MENDELEY_LIMIT = 100


def get_next_url(r_heads_dict):
	if 'Link' not in r_heads_dict:
		return None
	link_value = r_heads_dict['Link']
	next_field = next((x for x in link_value.split(',') if x.endswith('rel="next"')), None)
	if next_field:
		return next_field.rsplit(';',1)[0][1:-1] # remove <> symbols
	return None

def next_search(url):
	r = requests.get(url, headers=MENDELEY_HEADERS)
	batch_results = r.json()
	r_heads_dict = dict(r.headers)
	next_url = get_next_url(r_heads_dict)
	return batch_results, next_url

def search(query):
	all_results = []
	print("Searching for query: {}".format(query))
	payload = {
		'query': query,
		# 'title': title_query,
		# 'abstract': abstract_query,
		'limit': MENDELEY_LIMIT
	}
	r = requests.get(MENDELEY_API_BASE, headers=MENDELEY_HEADERS, params=payload)
	batch_results = r.json()

	if batch_results and (not isinstance(batch_results,list) or 'title' not in batch_results[0]):
		print('An error has occurred:')
		print(json.dumps(batch_results, indent=3, ensure_ascii=False))
		return

	if len(batch_results)==0:
		return batch_results
		# empty results

	r_heads_dict = dict(r.headers)
	total_count = r_heads_dict.get('Mendeley-Count', len(batch_results)) 
	# count not present if all return in first call    
	num_call = 1
	print('Total elements: {}'.format(total_count))
	print('Call {}: retrieving {} elements'.format(num_call, len(batch_results)))
	all_results.extend(batch_results)
	next_url = get_next_url(r_heads_dict)
	while next_url:
		num_call += 1
		next_search(next_url)
		batch_results, next_url = next_search(next_url)
		all_results.extend(batch_results)
		print('Call {}: retrieving {} elements'.format(num_call, len(batch_results)))    
	print('Total saved elements: {}'.format(len(all_results)))
	return all_results

def main_search():     
	query_counts = {}
	for q in MODS_SENSES_COMBO_STR:
		quoted_q = '"{}"'.format(q)
		results = search(quoted_q)        
		query_counts[q] = len(results)
		file_path = './data/mendeley/{}.json'.format(q)
		print('Writing to  {}'.format(file_path))
		with open(file_path, 'w') as f_out:
			json.dump(results, f_out, indent=3, ensure_ascii=False)
		print()
	# with open('./data/mendeley_stats.txt', 'w') as f_out:
	# 	sorted_query_counts = sorted(query_counts.items(), key=lambda x: -x[1])
	# 	for q,c in sorted_query_counts:
	# 		f_out.write('{}: {}\n'.format(q,c))

def get_doi(paper):
	if 'identifiers' in paper and 'doi' in paper['identifiers']:
		return paper['identifiers']['doi'].split()[0]
	return None

ID_KEY = 'id'
TITLE_KEY = 'title'
YEAR_KEY = 'year'
ABSTRACT_KEY = 'abstract'
KEYWORDS_KEY = 'keywords'

def normalize_entry(paper):
	return {
		'Source': 'Mendeley',
		'Source_id': paper.get(ID_KEY, None),
		'Year': paper.get(YEAR_KEY, None), 
		'Authors': ', '.join(
			['{} {}'.format(a.get('first_name', ''), a.get('last_name', '')).strip() 
			for a in paper.get('authors',[])]
		),
		'Title': paper[TITLE_KEY],
		'Abstract': paper.get(ABSTRACT_KEY, ''),
		'DOI': paper.get('identifiers', {}).get('doi',None)		
	}

def refine_results():
	from commons import refine_results
	return refine_results(
		'mendeley', 
		ID_KEY, TITLE_KEY, ABSTRACT_KEY, KEYWORDS_KEY,
		normalize_entry
	)

def year_distribution():
	import matplotlib.pyplot as plt; plt.rcdefaults()
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	records = json.load(open('./data/accepted_mendeley.json'))
	records_years = [r['year'] for r in records if 'year' in r and r['year']>1900]
	years = sorted(set(records_years))
	
	df = pd.DataFrame({
		'year':years,
		'freq':[records_years.count(y) for y in sorted(set(years))]
	})

	ax = df.plot(kind='bar',x='year',y='freq')
	ax.get_legend().remove()
	xticks = ax.xaxis.get_major_ticks()
	for i,tick in enumerate(xticks):
		if i%5 != 0:
			tick.label1.set_visible(False)

	plt.gcf().subplots_adjust(bottom=0.15)
	plt.show()


if __name__ == '__main__':
	# main_search()
	# refine_results()
	year_distribution()
	