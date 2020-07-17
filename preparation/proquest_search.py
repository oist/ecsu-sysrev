import os
import json
from commons import validate_paper
import xlrd

TITLE_KEY = 'Title'
ABSTRACT_KEY = 'Abstract'
AUTHORS_KEY = 'Authors'
YEAR_KEY = 'year'
DOI_KEY = 'digitalObjectIdentifier'


def get_papers_from_xls():    
    tsv_dir = 'data/proquest'
    for tsv_file in os.listdir(tsv_dir):
        if tsv_file.endswith('.xls'):
            tsv_file_path = os.path.join(tsv_dir, tsv_file)
            workbook = xlrd.open_workbook(tsv_file_path)
            worksheet = workbook.sheet_by_index(0)            
            headers = [str(h.value) for h in worksheet.row(0)]
            num_rows = len(worksheet.col(0))-1
            for i in range(1,num_rows):
                row = [str(c.value).strip() for c in worksheet.row(i)]
                assert len(row) == len(headers)
                yield {headers[i]:row[i] for i in range(len(row))}
                    

def get_doi(paper):
    return paper.get(DOI_KEY,None)

def normalize_entry(paper):
    return {
        'Source': 'Proquest',
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
    pbar = tqdm(total=14630)	
    all_papers, accepted_papers, accepted_papers_norm, discarded_papers  = [], [], [], []
    for p in get_papers_from_xls():
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
    with open('./data/proquest/proquest_all.json', 'w') as f_out:
        json.dump(all_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_proquest.json', 'w') as f_out:
        json.dump(accepted_papers, f_out, indent=3, ensure_ascii=False)
    with open('./data/accepted_proquest_norm.json', 'w') as f_out:
        json.dump(accepted_papers_norm, f_out, indent=3, ensure_ascii=False)
    with open('./data/discarded_proquest.json', 'w') as f_out:
        json.dump(discarded_papers, f_out, indent=3, ensure_ascii=False)

    discarded_papers = len(all_papers) - len(accepted_papers)
    print("Total papers: {}".format(len(all_papers)))
    print("Papers without abstract: {}".format(no_abstract_count))
    print("Discarded papers: {}".format(discarded_papers))
    print("Accepted papers: {}".format(len(accepted_papers)))
    return accepted_papers

if __name__ == '__main__':
    refine_results()    
    