import os
import json
import csv
from commons import validate_paper

TITLE_KEY = 'Title'
ABSTRACT_KEY = 'Abstract'
AUTHORS_KEY = '﻿Authors'
YEAR_KEY = 'Year'
DOI_KEY = 'DOI'


def get_papers_from_csv():    
    tsv_dir = 'data/scopus'
    for tsv_file in os.listdir(tsv_dir):
        # print(tsv_file)
        if tsv_file.endswith('csv'):
            tsv_file_path = os.path.join(tsv_dir, tsv_file)
            with open(tsv_file_path, encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                headers = next(reader)
                for row in reader:
                    assert len(row) == len(headers)
                    yield {headers[i]:row[i] for i in range(len(row))}
                    

def get_doi(paper):
    return paper.get(DOI_KEY,None)

def normalize_entry(paper):
    return {
        'Source': 'Scopus',
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
    pbar = tqdm(total=26877)	
    all_papers, accepted_papers, accepted_papers_norm, discarded_papers  = [], [], [], []
    for p in get_papers_from_csv():
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
    with open('./data/scopus/scopus_all.json', 'w') as f_out:
        json.dump(all_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_scopus.json', 'w') as f_out:
        json.dump(accepted_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_scopus_norm.json', 'w') as f_out:
        json.dump(accepted_papers_norm, f_out, indent=3, ensure_ascii=False)
    with open('./data/discarded_scopus.json', 'w') as f_out:
        json.dump(discarded_papers, f_out, indent=3, ensure_ascii=False)

    discarded_papers = len(all_papers) - len(accepted_papers)
    print("Total papers: {}".format(len(all_papers)))
    print("Papers without abstract: {}".format(no_abstract_count))
    print("Discarded papers: {}".format(discarded_papers))
    print("Accepted papers: {}".format(len(accepted_papers)))
    return accepted_papers

if __name__ == '__main__':
    refine_results()    
    