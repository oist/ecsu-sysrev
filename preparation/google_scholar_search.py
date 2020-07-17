# https://pypi.org/project/scholarly/

from commons import MODS_SENSES_COMBO_STR, validate_paper
from serpapi.google_search_results import GoogleSearchResults
from tqdm import tqdm
import json
import os
from key import GOOGLE_SCHOLAR_API_KEY

def google_serpapi_search(query):    

	all_results = []
	max_num_results = 20
	iter_num = 0

	def call_api():
		params = {
			"engine": "google_scholar",
			"q": query,
			"hl": "en",
			"as_sdt": "1",
			"api_key": GOOGLE_SCHOLAR_API_KEY,
			'start': iter_num*max_num_results,
			"num": max_num_results
		}
		client = GoogleSearchResults(params)
		return client.get_dict()
	
	results = call_api()
	if 'error' in results:
		error = results['error']
		if error == "Google hasn't returned any results for this query.":
			return []
		print('AN ERROR HAS OCCURED: {}'.format(error))
		return
	total_results = results['search_information']['total_results']
	pbar = tqdm(total=min(1000,total_results))	
	organic_results = results['organic_results']	
	pbar.update(len(organic_results))
	all_results.extend(organic_results)	
	while True:
		next_link = results['serpapi_pagination'].get('next', None) if 'serpapi_pagination' in results else None
		if next_link is None:
			break
		iter_num += 1
		results = call_api()
		if 'error' in results:
			print('AN ERROR HAS OCCURED: {}'.format(results['error']))
			return
		organic_results = results['organic_results']
		pbar.update(len(organic_results))
		all_results.extend(organic_results)
		# print(next_link)
	pbar.close()
	return all_results	

def scholarly_search(query):
	import scholarly	
	pbar = tqdm(total=1000)	
	search_query = scholarly.search_pubs_query(query)
	all_results = []
	for pub in search_query:
		all_results.append(pub.bib)
		pbar.update()
	pbar.close()
	return all_results

def main_search():     
	query_counts = {}
	for q in MODS_SENSES_COMBO_STR:
		quoted_q = '"{}"'.format(q)
		print('Searching  {}'.format(quoted_q))
		file_path = './data/google_scholar/{}.json'.format(q)
		if os.path.exists(file_path):
			print('Already present')
			with open(file_path) as f_in:
				results = json.load(f_in)			
		else:
			# results = scholarly_search(quoted_q)        
			results = google_serpapi_search(quoted_q)
		
		query_counts[q] = len(results)
		
		print("Number of results: {}".format(len(results)))
		print('Writing to  {}'.format(file_path))
		with open(file_path, 'w') as f_out:
			json.dump(results, f_out, indent=3, ensure_ascii=False)
		print()
	# with open('./data/google_scholar_stats.txt', 'w') as f_out:
	# 	sorted_query_counts = sorted(query_counts.items(), key=lambda x: -x[1])
	# 	for q,c in sorted_query_counts:
	# 		f_out.write('{}: {}\n'.format(q,c))

ID_KEY = 'result_id'
TITLE_KEY = 'title'
ABSTRACT_KEY = 'snippet'
KEYWORDS_KEY = None

def normalize_entry(paper):
	return {
		'Source': 'Google Scholar',
		'Source_id': paper.get(ID_KEY, None),
		'Year': paper.get('year', None), 
		'Authors': ', '.join(
			[a.get('name', '').strip() 
			for a in paper.get('publication_info',{}).get('authors', [])]
		),
		'Title': paper[TITLE_KEY],
		'Abstract': paper.get(ABSTRACT_KEY, None),
		'DOI': None,		
	}

def refine_results():
	from commons import refine_results
	return refine_results(
		'google_scholar', 
		ID_KEY, TITLE_KEY, ABSTRACT_KEY, KEYWORDS_KEY,
		normalize_entry
	)



if __name__ == '__main__':
	# google_search_result()
	# scholarly()
	# main_search()
	# google_serpapi_search('"passive acoustic"')
	refine_results()
	
	# q = '"dynamic smell"'
	# results = scholarly_search(q)
	# print(len(results))
