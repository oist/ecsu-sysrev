import os
import json
import csv
from commons import validate_paper

TITLE_KEY = 'TI'
ABSTRACT_KEY = 'AB'
AUTHORS_KEY = 'AU'
YEAR_KEY = 'PY'
DOI_KEY = 'DI'

def get_papers_from_manual_txt():
    with open('./data/wos/wos_manual.txt') as f_in:
        result = {}
        last_key = None
        for line in f_in:
            line_strip = line.strip()
            if line_strip=='':
                if result:
                    yield result
                    result = {}
            if line.startswith(' '):
                result[last_key] += ' ' + line_strip
            else:
                if ' ' in line_strip:
                    k,v = line_strip.split(' ', 1)
                    result[k] = v
                    last_key = k
                elif line_strip!='':
                    result[line_strip] = ''
        if result:
            yield result

def get_papers_from_tsv():
    tsv_dir = 'data/wos'
    for tsv_file in os.listdir(tsv_dir):
        if tsv_file.endswith('txt'):
            tsv_file_path = os.path.join(tsv_dir, tsv_file)
            with open(tsv_file_path, encoding='utf-16') as f_in:
                reader = csv.reader(f_in, delimiter='\t')
                headers = next(reader)
                for row in reader:
                    assert len(row) == len(headers)
                    yield {headers[i]:row[i] for i in range(len(row))}

def get_doi(paper):
    return paper.get(DOI_KEY,None)

def normalize_entry(paper):
    return {
        'Source': 'Web of Science',
        'Source_id': None,
        'Year': int(paper[YEAR_KEY]) if YEAR_KEY in paper and paper[YEAR_KEY] else None, 
        'Authors': paper.get(AUTHORS_KEY, None),
        'Title': paper[TITLE_KEY],
        'Abstract': paper.get(ABSTRACT_KEY, None),
        'DOI': paper.get(DOI_KEY, None)
    }

def refine_results():
    from tqdm import tqdm        
    no_abstract_count = 0
    pbar = tqdm(total=28195)	
    all_papers, accepted_papers, accepted_papers_norm, discarded_papers  = [], [], [], []
    for p in get_papers_from_tsv(): # get_papers_from_manual_txt():
        pbar.update()
        all_papers.append(p)
        if ABSTRACT_KEY not in p:
            no_abstract_count += 1
        title = p[TITLE_KEY]
        abstract = p.get(ABSTRACT_KEY, '')
        mod_on_matches, mod_off_matches, mod_sense_matches = validate_paper(title, abstract)
        if mod_on_matches and mod_off_matches and mod_sense_matches:
            accepted_papers.append(p)
            norm_p = normalize_entry(p)
            norm_p['mod_on_matches'] = mod_on_matches
            norm_p['mod_off_matches'] = mod_off_matches
            norm_p['mod_sense_matches'] = mod_sense_matches
            accepted_papers_norm.append(norm_p)
        else:
            discarded_papers.append(p)
    pbar.close()
    with open('./data/wos/wos_all.json', 'w') as f_out:
        json.dump(all_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_wos.json', 'w') as f_out:
        json.dump(accepted_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_wos_norm.json', 'w') as f_out:
        json.dump(accepted_papers_norm, f_out, indent=3, ensure_ascii=False)
    with open('./data/discarded_wos.json', 'w') as f_out:
        json.dump(discarded_papers, f_out, indent=3, ensure_ascii=False)

    discarded_papers = len(all_papers) - len(accepted_papers)
    print("Total papers: {}".format(len(all_papers)))
    print("Papers without abstract: {}".format(no_abstract_count))
    print("Discarded papers: {}".format(discarded_papers))
    print("Accepted papers: {}".format(len(accepted_papers)))
    return accepted_papers

def generate_search_query():
    from collections import defaultdict
    mendeley_hits_ge_1 = []
    with open('data/mendeley_stats.json') as f_in:
        query_info = json.load(f_in)
        for query, info in query_info.items():
            if int(info['TOTAL'])>0:
            	mendeley_hits_ge_1.append(query)            
    print(' OR '.join('"{}"'.format(x) for x in mendeley_hits_ge_1))
    print(len(mendeley_hits_ge_1))    	

if __name__ == '__main__':
    refine_results()
    # generate_search_query()
    